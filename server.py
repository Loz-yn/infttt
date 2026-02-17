from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room
import uuid
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ttt_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

DB_FILE = 'ttt_users.json'

games = {}
waiting_room = []
player_usernames = {}
active_sessions = {}


# ── DATABASE ─────────────────────────────────────────────────────────────────

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
    users[username] = { 'password': generate_password_hash(password), 'wins': 0, 'losses': 0, 'draws': 0 }
    save_users(users)
    return True

def authenticate_user(username, password):
    user = get_user(username)
    if user and check_password_hash(user['password'], password):
        return { 'username': username, 'wins': user['wins'], 'losses': user['losses'], 'draws': user['draws'] }
    return None

def update_user_stats(username, result):
    users = load_users()
    if username in users:
        users[username][result + 's'] += 1
        save_users(users)
        return { 'wins': users[username]['wins'], 'losses': users[username]['losses'], 'draws': users[username]['draws'] }
    return None

def get_leaderboard():
    users = load_users()
    lb = [{ 'username': u, **{k: v for k, v in d.items() if k != 'password'} } for u, d in users.items()]
    lb.sort(key=lambda x: (x['wins'], x['wins'] / max(x['wins'] + x['losses'] + x['draws'], 1)), reverse=True)
    return lb


# ── GAME CLASS ────────────────────────────────────────────────────────────────

WIN_LINES = [
    [0,1,2],[3,4,5],[6,7,8],
    [0,3,6],[1,4,7],[2,5,8],
    [0,4,8],[2,4,6]
]

class TTTGame:
    def __init__(self, game_id):
        self.id          = game_id
        self.board       = [None] * 9
        self.move_order  = []
        self.turn        = 'X'
        self.x_player    = None
        self.o_player    = None
        self.x_username  = None
        self.o_username  = None

    def make_move(self, index, mark):
        if self.turn != mark:
            return False, None, False, None, None
        if self.board[index] is not None:
            return False, None, False, None, None

        removed_index = None
        my_pieces = [(m, i) for m, i in self.move_order if m == mark]

        if len(my_pieces) >= 3:
            _, oldest_index = my_pieces[0]
            self.board[oldest_index] = None
            self.move_order = [(m, i) for m, i in self.move_order if not (m == mark and i == oldest_index)]
            removed_index = oldest_index

        self.board[index] = mark
        self.move_order.append((mark, index))

        win_line = self._check_winner(mark)
        if win_line:
            return True, removed_index, True, mark, win_line

        self.turn = 'O' if mark == 'X' else 'X'
        return True, removed_index, False, None, None

    def _check_winner(self, mark):
        for line in WIN_LINES:
            if all(self.board[i] == mark for i in line):
                return line
        return None


# ── SOCKET HANDLERS ───────────────────────────────────────────────────────────

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
        emit('login_success', {
            'username': username,
            'stats': { 'wins': user_data['wins'], 'losses': user_data['losses'], 'draws': user_data['draws'] }
        })
    else:
        user = get_user(username)
        if user:
            emit('login_failed', {'message': 'Incorrect password!'})
        else:
            if create_user(username, password):
                active_sessions[request.sid] = username
                emit('login_success', { 'username': username, 'stats': { 'wins': 0, 'losses': 0, 'draws': 0 } })
            else:
                emit('login_failed', {'message': 'Failed to create account!'})


@socketio.on('find_game')
def handle_find_game(data):
    player_id = request.sid
    username  = data.get('username', 'Anonymous')
    player_usernames[player_id] = username

    if waiting_room:
        opp_id       = waiting_room.pop(0)
        opp_username = player_usernames.get(opp_id, 'Anonymous')
        game_id      = str(uuid.uuid4())

        game            = TTTGame(game_id)
        game.x_player   = opp_id
        game.o_player   = player_id
        game.x_username = opp_username
        game.o_username = username
        games[game_id]  = game

        join_room(game_id, sid=opp_id)
        join_room(game_id, sid=player_id)

        emit('game_start', {
            'game_id': game_id, 'mark': 'X', 'first_turn': 'X',
            'opponent_name': username,
            'x_player': opp_username, 'o_player': username,
        }, room=opp_id)

        emit('game_start', {
            'game_id': game_id, 'mark': 'O', 'first_turn': 'X',
            'opponent_name': opp_username,
            'x_player': opp_username, 'o_player': username,
        }, room=player_id)
    else:
        waiting_room.append(player_id)
        emit('waiting', {'message': 'Waiting for opponent...'})


@socketio.on('make_move')
def handle_make_move(data):
    game_id   = data.get('game_id')
    index     = data.get('index')
    player_id = request.sid

    if game_id not in games:
        return

    game = games[game_id]
    mark = 'X' if game.x_player == player_id else 'O'

    success, removed_index, game_over, winner, win_line = game.make_move(index, mark)

    if not success:
        emit('error', {'message': 'Invalid move!'})
        return

    stats_x = stats_o = None
    if game_over and winner:
        stats_x = update_user_stats(game.x_username, 'win' if winner == 'X' else 'loss')
        stats_o = update_user_stats(game.o_username, 'win' if winner == 'O' else 'loss')

    emit('move_made', {
        'index': index, 'mark': mark,
        'removed_index': removed_index,
        'game_over': game_over, 'winner': winner, 'win_line': win_line,
        'stats': stats_x
    }, room=game.x_player)

    emit('move_made', {
        'index': index, 'mark': mark,
        'removed_index': removed_index,
        'game_over': game_over, 'winner': winner, 'win_line': win_line,
        'stats': stats_o
    }, room=game.o_player)

    if game_over:
        del games[game_id]


@socketio.on('chat_message')
def handle_chat(data):
    game_id = data.get('game_id')
    message = data.get('message', '').strip()
    sender  = data.get('sender', 'Anonymous')
    if not message or len(message) > 200 or game_id not in games:
        return
    emit('chat_message', { 'sender': sender, 'message': message }, room=game_id, skip_sid=request.sid)


@socketio.on('get_leaderboard')
def handle_leaderboard():
    emit('leaderboard_data', { 'leaderboard': get_leaderboard() })


@socketio.on('disconnect')
def handle_disconnect():
    player_id = request.sid
    if player_id in waiting_room:
        waiting_room.remove(player_id)
    player_usernames.pop(player_id, None)
    active_sessions.pop(player_id, None)

    for game_id, game in list(games.items()):
        if player_id in (game.x_player, game.o_player):
            emit('opponent_disconnected', room=game_id, skip_sid=player_id)
            del games[game_id]
            break


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)