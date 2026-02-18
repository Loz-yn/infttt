from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room
import uuid, json, os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ttt_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

DB_FILE = 'ttt_users.json'
games = {}
waiting_room = []
player_usernames = {}
active_sessions = {}
rematch_requests = {}  # game_id -> set of player sids who requested rematch


# ── DATABASE ──────────────────────────────────────────────────────────────────

def load_users():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def get_user(username):
    return load_users().get(username)


def create_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = {'password': generate_password_hash(password), 'wins': 0, 'losses': 0, 'draws': 0}
    save_users(users)
    return True


def authenticate_user(username, password):
    user = get_user(username)
    if user and check_password_hash(user['password'], password):
        return {'username': username, 'wins': user['wins'], 'losses': user['losses'], 'draws': user['draws']}
    return None


def update_user_stats(username, result):
    key_map = {'win': 'wins', 'loss': 'losses', 'draw': 'draws'}
    users = load_users()
    if username in users:
        users[username][key_map[result]] += 1
        save_users(users)
        return {'wins': users[username]['wins'], 'losses': users[username]['losses'], 'draws': users[username]['draws']}
    return None


def get_leaderboard():
    users = load_users()
    lb = [{'username': u, **{k: v for k, v in d.items() if k != 'password'}} for u, d in users.items()]
    lb.sort(key=lambda x: (x['wins'], x['wins'] / max(x['wins'] + x['losses'] + x['draws'], 1)), reverse=True)
    return lb


# ── GAME CLASS ────────────────────────────────────────────────────────────────

WIN_LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
MATCH_WINS_NEEDED = 2


class TTTGame:
    def __init__(self, game_id):
        self.id = game_id
        self.board = [None] * 9
        self.move_order = []
        self.turn = 'X'
        self.x_player = None
        self.o_player = None
        self.x_username = None
        self.o_username = None
        self.match_score = {'X': 0, 'O': 0}
        self.round = 1

    def make_move(self, index, mark):
        if self.turn != mark or self.board[index] is not None:
            return False, None, False, None, None

        removed_index = None
        my_pieces = [(m, i) for m, i in self.move_order if m == mark]

        # Step 1: Determine which piece would be removed (the oldest)
        oldest_index = None
        if len(my_pieces) >= 3:
            oldest_index = my_pieces[0][1]

        # Step 2: Simulate the final board state —
        # remove oldest, place new piece, then check win
        temp_board = self.board[:]
        if oldest_index is not None:
            temp_board[oldest_index] = None
        temp_board[index] = mark

        win_line = None
        for line in WIN_LINES:
            if all(temp_board[i] == mark for i in line):
                win_line = line
                break

        # Step 3: Apply the changes for real
        if oldest_index is not None:
            self.board[oldest_index] = None
            self.move_order = [(m, i) for m, i in self.move_order
                               if not (m == mark and i == oldest_index)]
            removed_index = oldest_index

        self.board[index] = mark
        self.move_order.append((mark, index))

        if win_line:
            return True, removed_index, True, mark, win_line

        self.turn = 'O' if mark == 'X' else 'X'
        return True, removed_index, False, None, None

    def _check_winner(self, mark):
        for line in WIN_LINES:
            if all(self.board[i] == mark for i in line):
                return line
        return None

    def reset_round(self):
        self.board = [None] * 9
        self.move_order = []
        self.turn = 'X'
        self.round += 1

    def get_opponent(self, player_id):
        return self.o_player if player_id == self.x_player else self.x_player

    def get_mark(self, player_id):
        return 'X' if player_id == self.x_player else 'O'


# ── AUTH HANDLERS ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('login')
def handle_login(data):
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    if not username or not password:
        emit('login_failed', {'message': 'Username and password required!'})
        return
    user_data = authenticate_user(username, password)
    if user_data:
        active_sessions[request.sid] = username
        emit('login_success', {'username': username, 'stats': {'wins': user_data['wins'], 'losses': user_data['losses'],
                                                               'draws': user_data['draws']}})
    else:
        user = get_user(username)
        if user:
            emit('login_failed', {'message': 'Incorrect password!'})
        else:
            if create_user(username, password):
                active_sessions[request.sid] = username
                emit('login_success', {'username': username, 'stats': {'wins': 0, 'losses': 0, 'draws': 0}})
            else:
                emit('login_failed', {'message': 'Failed to create account!'})


# ── MATCHMAKING ───────────────────────────────────────────────────────────────

@socketio.on('find_game')
def handle_find_game(data):
    player_id = request.sid
    username = data.get('username', 'Anonymous')
    player_usernames[player_id] = username

    if waiting_room:
        opp_id = waiting_room.pop(0)
        opp_username = player_usernames.get(opp_id, 'Anonymous')
        game_id = str(uuid.uuid4())

        game = TTTGame(game_id)
        game.x_player = opp_id
        game.o_player = player_id
        game.x_username = opp_username
        game.o_username = username
        games[game_id] = game

        join_room(game_id, sid=opp_id)
        join_room(game_id, sid=player_id)

        emit('game_start', {'game_id': game_id, 'mark': 'X', 'first_turn': 'X', 'new_match': True,
                            'opponent_name': username, 'x_player': opp_username, 'o_player': username}, room=opp_id)
        emit('game_start', {'game_id': game_id, 'mark': 'O', 'first_turn': 'X', 'new_match': True,
                            'opponent_name': opp_username, 'x_player': opp_username, 'o_player': username},
             room=player_id)
    else:
        waiting_room.append(player_id)
        emit('waiting', {'message': 'Waiting for opponent...'})


# ── MOVES ─────────────────────────────────────────────────────────────────────

@socketio.on('make_move')
def handle_make_move(data):
    game_id = data.get('game_id')
    index = data.get('index')
    player_id = request.sid

    print(f"[make_move] game={game_id} index={index} player={player_id}", flush=True)

    if game_id not in games:
        print(f"[make_move] game not found!", flush=True)
        emit('error', {'message': 'Game not found'})
        return

    try:
        game = games[game_id]
        mark = game.get_mark(player_id)

        print(f"[make_move] mark={mark} turn={game.turn} board={game.board}", flush=True)

        success, removed_index, game_over, winner, win_line = game.make_move(index, mark)

        print(f"[make_move] success={success} game_over={game_over} winner={winner} win_line={win_line}", flush=True)

        if not success:
            emit('error', {'message': 'Invalid move!'})
            return

        stats_x = stats_o = None
        if game_over and winner:
            game.match_score[winner] += 1
            stats_x = update_user_stats(game.x_username, 'win' if winner == 'X' else 'loss')
            stats_o = update_user_stats(game.o_username, 'win' if winner == 'O' else 'loss')
            print(f"[make_move] stats updated x={stats_x} o={stats_o}", flush=True)

        payload_base = {'index': index, 'mark': mark, 'removed_index': removed_index,
                        'game_over': game_over, 'winner': winner,
                        'win_line': list(win_line) if win_line else None}

        emit('move_made', {**payload_base, 'stats': stats_x}, room=game.x_player)
        emit('move_made', {**payload_base, 'stats': stats_o}, room=game.o_player)
        print(f"[make_move] emitted OK", flush=True)

        # If round ended but match not over, auto-start next round after short delay
        if game_over:
            match_over = game.match_score['X'] >= MATCH_WINS_NEEDED or game.match_score['O'] >= MATCH_WINS_NEEDED
            if not match_over:
                socketio.sleep(1.5)
                _start_next_round(game_id)
            # If match IS over, game stays until players leave/rematch manually

    except Exception as e:
        import traceback
        print(f"[make_move] EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        emit('error', {'message': f'Server error: {str(e)}'})


def _start_next_round(game_id):
    """Auto-start the next round for an ongoing match."""
    print(f"[next_round] called for game {game_id}", flush=True)

    if game_id not in games:
        print(f"[next_round] game {game_id} already gone", flush=True)
        return

    game = games[game_id]
    game.reset_round()

    first = 'X' if game.round % 2 == 1 else 'O'
    new_game_id = str(uuid.uuid4())

    print(f"[next_round] creating new game {new_game_id} round={game.round} first={first}", flush=True)

    new_game = TTTGame(new_game_id)
    new_game.x_player = game.x_player
    new_game.o_player = game.o_player
    new_game.x_username = game.x_username
    new_game.o_username = game.o_username
    new_game.match_score = dict(game.match_score)
    new_game.round = game.round
    new_game.turn = first  # SET THE CORRECT STARTING TURN
    games[new_game_id] = new_game

    print(f"[next_round] new game turn={new_game.turn} emitting to X sid={game.x_player}", flush=True)
    socketio.emit('rematch_start', {
        'game_id': new_game_id, 'mark': 'X', 'first_turn': first, 'new_match': False,
        'opponent_name': game.o_username, 'x_player': game.x_username, 'o_player': game.o_username
    }, room=game.x_player, namespace='/')

    print(f"[next_round] emitting to O sid={game.o_player}", flush=True)
    socketio.emit('rematch_start', {
        'game_id': new_game_id, 'mark': 'O', 'first_turn': first, 'new_match': False,
        'opponent_name': game.x_username, 'x_player': game.x_username, 'o_player': game.o_username
    }, room=game.o_player, namespace='/')

    del games[game_id]
    print(f"[next_round] old game deleted", flush=True)


@socketio.on('timeout')
def handle_timeout(data):
    """Player ran out of time — make a random move for them."""
    game_id = data.get('game_id')
    player_id = request.sid

    print(f"[timeout] player={player_id} game={game_id}", flush=True)

    if game_id not in games:
        print(f"[timeout] game not found", flush=True)
        return

    game = games[game_id]
    mark = game.get_mark(player_id)

    # Only the player whose turn it is can time out
    if game.turn != mark:
        print(f"[timeout] not this player's turn", flush=True)
        return

    # Find all empty cells
    empty_cells = [i for i in range(9) if game.board[i] is None]

    if not empty_cells:
        print(f"[timeout] no empty cells!", flush=True)
        return

    # Pick a random empty cell
    import random
    random_index = random.choice(empty_cells)

    print(f"[timeout] auto-placing {mark} at {random_index}", flush=True)

    # Make the move (use the same logic as handle_make_move)
    success, removed_index, game_over, winner, win_line = game.make_move(random_index, mark)

    if not success:
        print(f"[timeout] move failed!", flush=True)
        return

    stats_x = stats_o = None
    if game_over and winner:
        game.match_score[winner] += 1
        stats_x = update_user_stats(game.x_username, 'win' if winner == 'X' else 'loss')
        stats_o = update_user_stats(game.o_username, 'win' if winner == 'O' else 'loss')

    payload_base = {
        'index': random_index,
        'mark': mark,
        'removed_index': removed_index,
        'game_over': game_over,
        'winner': winner,
        'win_line': list(win_line) if win_line else None,
        'timeout': True  # Signal this was an auto-move
    }

    socketio.emit('move_made', {**payload_base, 'stats': stats_x}, room=game.x_player, namespace='/')
    socketio.emit('move_made', {**payload_base, 'stats': stats_o}, room=game.o_player, namespace='/')

    # If round ended but match not over, auto-start next round
    if game_over:
        match_over = game.match_score['X'] >= MATCH_WINS_NEEDED or game.match_score['O'] >= MATCH_WINS_NEEDED
        if not match_over:
            socketio.sleep(1.5)
            _start_next_round(game_id)


# ── REMATCH ───────────────────────────────────────────────────────────────────

@socketio.on('rematch_request')
def handle_rematch(data):
    game_id = data.get('game_id')
    player_id = request.sid

    if game_id not in games:
        return

    game = games[game_id]

    if game_id not in rematch_requests:
        rematch_requests[game_id] = set()

    rematch_requests[game_id].add(player_id)

    # Both players agreed — start next round
    if len(rematch_requests[game_id]) == 2:
        del rematch_requests[game_id]

        match_over = game.match_score['X'] >= MATCH_WINS_NEEDED or game.match_score['O'] >= MATCH_WINS_NEEDED
        new_match = match_over

        if new_match:
            game.match_score = {'X': 0, 'O': 0}
            game.round = 0

        game.reset_round()

        # Swap who goes first each round
        first = 'X' if game.round % 2 == 1 else 'O'
        new_game_id = str(uuid.uuid4())

        # Create a continuation game with same players
        new_game = TTTGame(new_game_id)
        new_game.x_player = game.x_player
        new_game.o_player = game.o_player
        new_game.x_username = game.x_username
        new_game.o_username = game.o_username
        new_game.match_score = game.match_score
        new_game.round = game.round
        games[new_game_id] = new_game

        join_room(new_game_id, sid=game.x_player)
        join_room(new_game_id, sid=game.o_player)

        emit('rematch_start', {'game_id': new_game_id, 'mark': 'X', 'first_turn': first, 'new_match': new_match,
                               'opponent_name': game.o_username, 'x_player': game.x_username,
                               'o_player': game.o_username}, room=game.x_player)
        emit('rematch_start', {'game_id': new_game_id, 'mark': 'O', 'first_turn': first, 'new_match': new_match,
                               'opponent_name': game.x_username, 'x_player': game.x_username,
                               'o_player': game.o_username}, room=game.o_player)

        del games[game_id]


@socketio.on('leave_game')
def handle_leave_game(data):
    """Player leaves from end screen — cancel any rematch and clean up."""
    game_id = data.get('game_id')
    player_id = request.sid

    print(f"[leave_game] player={player_id} game={game_id}", flush=True)

    if not game_id:
        return

    # Cancel any pending rematch request
    if game_id in rematch_requests:
        rematch_requests[game_id].discard(player_id)
        # If opponent is waiting, tell them the rematch is declined
        if len(rematch_requests[game_id]) > 0:
            print(f"[leave_game] notifying opponent", flush=True)
            # Get opponent's socket ID
            if game_id in games:
                game = games[game_id]
                opp = game.get_opponent(player_id)
                socketio.emit('rematch_declined', {}, room=opp, namespace='/')
        del rematch_requests[game_id]

    # Clean up game
    if game_id in games:
        del games[game_id]
        print(f"[leave_game] game {game_id} deleted", flush=True)


# ── CHAT ──────────────────────────────────────────────────────────────────────

@socketio.on('chat_message')
def handle_chat(data):
    game_id = data.get('game_id')
    message = data.get('message', '').strip()
    sender = data.get('sender', 'Anonymous')
    if not message or len(message) > 200 or game_id not in games:
        return
    emit('chat_message', {'sender': sender, 'message': message}, room=game_id, skip_sid=request.sid)


@socketio.on('get_leaderboard')
def handle_leaderboard():
    emit('leaderboard_data', {'leaderboard': get_leaderboard()})


# ── DISCONNECT ────────────────────────────────────────────────────────────────

@socketio.on('disconnect')
def handle_disconnect():
    player_id = request.sid
    if player_id in waiting_room:
        waiting_room.remove(player_id)
    player_usernames.pop(player_id, None)
    active_sessions.pop(player_id, None)

    for game_id, game in list(games.items()):
        if player_id in (game.x_player, game.o_player):
            opp = game.get_opponent(player_id)
            emit('opponent_disconnected', room=game_id, skip_sid=player_id)
            # Tell opponent their rematch request is dead
            if game_id in rematch_requests:
                emit('rematch_declined', room=opp)
                del rematch_requests[game_id]
            del games[game_id]
            break


if __name__ == '__main__':
    import os

    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)