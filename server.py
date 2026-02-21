from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room
import uuid, json, os
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
import psycopg2.extras

app = Flask(__name__)
app.config['SECRET_KEY'] = "ttt-secretkey"
socketio = SocketIO(app, cors_allowed_origins=os.environ.get('ALLOWED_ORIGIN', '*'), async_mode='gevent', manage_session=False)

games = {}
waiting_casual = []        # queue for casual (no MMR change)
waiting_ranked = []        # queue for ranked (MMR change)
player_usernames = {}
active_sessions = {}       # sid -> username
username_to_sid = {}       # username -> sid (enforces single active session per user)
rematch_requests = {}      # game_id -> set of player sids who requested rematch


# ── DATABASE ──────────────────────────────────────────────────────────────────

# Rank floor: the minimum MMR a player can fall to once they've reached a tier.
# Floors sit at the entry threshold of each tier's lowest sub-rank.
RANK_FLOORS = [
    (4500, 4500),   # Apex floor (can't fall out of Apex)
    (3000, 3000),   # Onyx floor (Bronze 1 entry → Onyx 3 @ 2700... floor at Onyx entry 3000? No — floor at Onyx 3)
    (2700, 2700),   # Onyx 3 is the entry to Onyx tier
    (2250, 2250),   # Obsidian 3 is the entry to Obsidian tier
    (1900, 1900),   # Ruby 3 is the entry to Ruby tier
    (1600, 1600),   # Gold 3 is the entry to Gold tier
    (1300, 1300),   # Silver 3 is the entry to Silver tier
    (1100, 1100),   # Bronze 2 is the entry to Bronze tier (above starting Bronze 3)
    (0, 0),         # Bronze 3 — absolute floor
]


def get_rank_floor(mmr):
    """Return the MMR floor for the current tier at the given MMR."""
    # Tiers and their floor (entry MMR of the lowest sub-rank in that tier)
    tier_floors = [
        (4500, 4500),  # Apex
        (2700, 2700),  # Onyx  (Onyx 3 @ 2700)
        (2250, 2250),  # Obsidian (Obsidian 3 @ 2250)
        (1900, 1900),  # Ruby (Ruby 3 @ 1900)
        (1600, 1600),  # Gold (Gold 3 @ 1600)
        (1300, 1300),  # Silver (Silver 3 @ 1300)
        (1100, 1100),  # Bronze upper (Bronze 2 @ 1100)
        (0, 0),        # Bronze 3 — no floor
    ]
    for threshold, floor in tier_floors:
        if mmr >= threshold:
            return floor
    return 0


def get_rank_name(mmr):
    """Convert MMR to rank name"""
    ranks = [
        ('Apex', 4500),
        ('Onyx 1', 3000),
        ('Onyx 2', 2850),
        ('Onyx 3', 2700),
        ('Obsidian 1', 2550),
        ('Obsidian 2', 2400),
        ('Obsidian 3', 2250),
        ('Ruby 1', 2100),
        ('Ruby 2', 2000),
        ('Ruby 3', 1900),
        ('Gold 1', 1800),
        ('Gold 2', 1700),
        ('Gold 3', 1600),
        ('Silver 1', 1500),
        ('Silver 2', 1400),
        ('Silver 3', 1300),
        ('Bronze 1', 1200),
        ('Bronze 2', 1100),
        ('Bronze 3', 0)
    ]
    for name, threshold in ranks:
        if mmr >= threshold:
            return name
    return 'Bronze 3'


def get_conn():
    url = os.environ['DATABASE_URL']
    # Railway uses postgres:// but psycopg2 needs postgresql://
    if url.startswith('postgres://'):
        url = url.replace('postgres://', 'postgresql://', 1)
    return psycopg2.connect(url)


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Create users table with rank instead of draws
            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    rank INTEGER DEFAULT 1000
                )
            ''')
            # Migrate existing users: remove draws column if it exists, add rank if missing
            cur.execute('''
                DO $$ 
                BEGIN
                    -- Drop draws column if it exists
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = 'users' AND column_name = 'draws'
                    ) THEN
                        ALTER TABLE users DROP COLUMN draws;
                    END IF;

                    -- Add rank column if it doesn't exist
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = 'users' AND column_name = 'rank'
                    ) THEN
                        ALTER TABLE users ADD COLUMN rank INTEGER DEFAULT 1000;
                    END IF;

                    -- Add demotion_shield column if it doesn't exist
                    -- 3 charges protect you at a tier floor before you derank
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = 'users' AND column_name = 'demotion_shield'
                    ) THEN
                        ALTER TABLE users ADD COLUMN demotion_shield INTEGER DEFAULT 3;
                    END IF;
                END $$;
            ''')
        conn.commit()


def get_user(username):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            return cur.fetchone()


def create_user(username, password):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, password) VALUES (%s, %s)",
                    (username, generate_password_hash(password))
                )
            conn.commit()
        return True
    except psycopg2.errors.UniqueViolation:
        return False


def authenticate_user(username, password):
    user = get_user(username)
    if user and check_password_hash(user['password'], password):
        return {'username': username, 'wins': user['wins'], 'losses': user['losses'],
                'rank': user['rank'], 'demotion_shield': user.get('demotion_shield', 3)}
    return None


def get_k_factor(rank, total_games):
    """
    Dynamic K-factor based on rank and experience.
    - New players (< 10 games): K=40 so they settle quickly
    - Low rank (< 1200): K=24 — meaningful but not punishing
    - Mid rank (1200–2100): K=20 — standard competitive
    - High rank (2100–2700): K=16 — slower movement, harder to climb/fall
    - Elite rank (>= 2700): K=12 — very stable at the top
    """
    if total_games < 10:
        return 40
    if rank >= 2700:
        return 12
    if rank >= 2100:
        return 16
    if rank >= 1200:
        return 20
    return 24


DEMOTION_SHIELD_MAX = 3  # Max shield charges per tier floor


def update_user_stats(username, result, opponent_username=None):
    """
    Update user stats with ELO + demotion shield system.

    Demotion shield:
    - Each tier floor (e.g. 1300 Silver, 1900 Ruby, 2700 Onyx, 4500 Apex) has 3 shield charges.
    - When you lose AT the exact floor MMR, a shield charge absorbs the loss instead of dropping you.
    - Once all charges are gone, losses derank you normally (dropping below the floor into the tier below).
    - Winning at the floor restores 1 charge (up to the max).
    - Reaching a NEW, higher floor for the first time resets your shield to full (3 charges).

    result: 'win' or 'loss'
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            if result == 'win':
                cur.execute("UPDATE users SET wins = wins + 1 WHERE username = %s", (username,))
            else:
                cur.execute("UPDATE users SET losses = losses + 1 WHERE username = %s", (username,))

            if opponent_username:
                cur.execute("SELECT rank, wins, losses, demotion_shield FROM users WHERE username = %s FOR UPDATE", (username,))
                row = cur.fetchone()
                user_rank, user_wins, user_losses, shield = row
                shield = shield if shield is not None else DEMOTION_SHIELD_MAX
                total_games = user_wins + user_losses

                cur.execute("SELECT rank FROM users WHERE username = %s", (opponent_username,))
                opp_rank = cur.fetchone()[0]

                # Dynamic K-factor
                k = get_k_factor(user_rank, total_games)

                # Standard ELO
                expected_score = 1 / (1 + 10 ** ((opp_rank - user_rank) / 400))
                actual_score   = 1.0 if result == 'win' else 0.0
                rank_change    = k * (actual_score - expected_score)

                # Upset bonus / protection
                if result == 'win' and opp_rank - user_rank >= 300:
                    rank_change *= 1.2
                elif result == 'loss' and user_rank - opp_rank >= 300:
                    rank_change = max(rank_change, -k * 0.5)

                rank_change = int(rank_change)
                rank_floor  = get_rank_floor(user_rank)
                at_floor    = (user_rank <= rank_floor)  # sitting exactly at or below the tier floor

                shield_event = None  # will be sent to client: 'blocked', 'restored', 'broken', None

                if result == 'loss' and at_floor:
                    if shield > 0:
                        # Shield absorbs the loss — MMR unchanged, consume one charge
                        shield -= 1
                        rank_change = 0
                        shield_event = 'blocked'
                    else:
                        # Shield depleted — derank normally (no floor protection this time)
                        shield_event = 'broken'
                        # rank_change is already negative, apply it freely (no floor clamp)
                        cur.execute(
                            "UPDATE users SET rank = GREATEST(0, rank + %s), demotion_shield = %s WHERE username = %s",
                            (rank_change, DEMOTION_SHIELD_MAX, username)  # reset shield for the new (lower) tier
                        )
                        conn.commit()
                        user = get_user(username)
                        return {
                            'wins': user['wins'], 'losses': user['losses'],
                            'rank': user['rank'], 'demotion_shield': user.get('demotion_shield', DEMOTION_SHIELD_MAX),
                            'shield_event': shield_event
                        }

                elif result == 'win':
                    new_rank   = user_rank + rank_change
                    new_floor  = get_rank_floor(new_rank)
                    if new_floor > rank_floor:
                        # Climbed into a new tier — reset shield to full
                        shield = DEMOTION_SHIELD_MAX
                    elif at_floor and shield < DEMOTION_SHIELD_MAX:
                        # Won while sitting at the floor — restore 1 charge
                        shield = min(DEMOTION_SHIELD_MAX, shield + 1)
                        shield_event = 'restored'

                    cur.execute(
                        "UPDATE users SET rank = GREATEST(0, rank + %s), demotion_shield = %s WHERE username = %s",
                        (rank_change, shield, username)
                    )
                    conn.commit()
                    user = get_user(username)
                    return {
                        'wins': user['wins'], 'losses': user['losses'],
                        'rank': user['rank'], 'demotion_shield': user.get('demotion_shield', DEMOTION_SHIELD_MAX),
                        'shield_event': shield_event
                    }

                else:
                    # Normal loss not at floor — apply with floor clamp
                    cur.execute(
                        "UPDATE users SET rank = GREATEST(%s, rank + %s), demotion_shield = %s WHERE username = %s",
                        (rank_floor, rank_change, shield, username)
                    )

                cur.execute(
                    "UPDATE users SET demotion_shield = %s WHERE username = %s",
                    (shield, username)
                )

        conn.commit()

    user = get_user(username)
    return {
        'wins': user['wins'], 'losses': user['losses'],
        'rank': user['rank'], 'demotion_shield': user.get('demotion_shield', DEMOTION_SHIELD_MAX),
        'shield_event': shield_event
    }


def get_leaderboard():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT username, wins, losses, rank FROM users ORDER BY rank DESC, wins DESC LIMIT 100")
            return [dict(row) for row in cur.fetchall()]


# ── GAME CLASS ────────────────────────────────────────────────────────────────

WIN_LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
MATCH_WINS_NEEDED = 2


class TTTGame:
    def __init__(self, game_id, mode='ranked'):
        self.id = game_id
        self.mode = mode
        self.board = [None] * 9
        self.move_order = []
        self.turn = 'X'
        self.x_player = None
        self.o_player = None
        self.x_username = None
        self.o_username = None
        self.match_score = {'X': 0, 'O': 0}
        self.round = 1
        # SECURITY: Server-side timer for each turn
        import time
        self.turn_start_time = time.time()

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


def _kick_existing_session(username):
    """Disconnect any existing session for this username and clean up all state."""
    old_sid = username_to_sid.get(username)
    if not old_sid:
        return

    print(f"[login] kicking old session {old_sid} for {username}", flush=True)

    # Remove from either waiting queue
    for q in (waiting_casual, waiting_ranked):
        if old_sid in q:
            q.remove(old_sid)

    # If mid-game, forfeit and notify opponent
    for game_id, game in list(games.items()):
        if old_sid in (game.x_player, game.o_player):
            opp = game.get_opponent(old_sid)
            socketio.emit('opponent_disconnected', {}, room=opp, namespace='/')
            if game_id in rematch_requests:
                del rematch_requests[game_id]
            del games[game_id]
            break

    # Clean up session maps
    active_sessions.pop(old_sid, None)
    player_usernames.pop(old_sid, None)

    # Tell the old tab it's been logged out
    socketio.emit('force_logout', {'message': 'You were logged in from another tab or device.'}, room=old_sid, namespace='/')


@socketio.on('login')
def handle_login(data):
    username = data.get('username', '')
    password = data.get('password', '')
    if not isinstance(username, str) or not isinstance(password, str):
        emit('login_failed', {'message': 'Invalid input.'})
        return
    username = username.strip()
    password = password.strip()
    if not username or not password:
        emit('login_failed', {'message': 'Username and password required!'})
        return
    if len(username) > 20:
        emit('login_failed', {'message': 'Username too long (max 20 chars).'})
        return
    if len(password) > 72:
        # bcrypt silently truncates at 72 bytes; enforce here to prevent DoS
        emit('login_failed', {'message': 'Password too long (max 72 chars).'})
        return
    # Rate limit: max LOGIN_MAX attempts per LOGIN_WINDOW seconds per socket
    now = time.time()
    attempts = login_attempts.get(request.sid, [])
    attempts = [t for t in attempts if now - t < LOGIN_WINDOW]
    if len(attempts) >= LOGIN_MAX:
        emit('login_failed', {'message': f'Too many attempts. Wait {LOGIN_WINDOW}s.'})
        return
    attempts.append(now)
    login_attempts[request.sid] = attempts

    user_data = authenticate_user(username, password)
    if user_data:
        # Kick any existing session for this user before creating a new one
        _kick_existing_session(username)
        active_sessions[request.sid] = username
        username_to_sid[username] = request.sid
        emit('login_success', {'username': username, 'stats': {'wins': user_data['wins'], 'losses': user_data['losses'],
                                                               'rank': user_data['rank'], 'demotion_shield': user_data.get('demotion_shield', 3)}})
    else:
        user = get_user(username)
        if user:
            emit('login_failed', {'message': 'Invalid username or password.'})
        else:
            if create_user(username, password):
                _kick_existing_session(username)
                active_sessions[request.sid] = username
                username_to_sid[username] = request.sid
                emit('login_success', {'username': username, 'stats': {'wins': 0, 'losses': 0, 'rank': 1000, 'demotion_shield': 3}})
            else:
                emit('login_failed', {'message': 'Failed to create account!'})


# ── MATCHMAKING ───────────────────────────────────────────────────────────────

def _find_opponent(queue, username, player_id):
    opp_id = None
    while queue:
        candidate = queue.pop(0)
        if candidate == player_id:
            continue
        if active_sessions.get(candidate) == username:
            player_usernames.pop(candidate, None)
            continue
        if any(candidate in (g.x_player, g.o_player) for g in games.values()):
            continue
        if candidate not in active_sessions:
            player_usernames.pop(candidate, None)
            continue
        opp_id = candidate
        break
    return opp_id


@socketio.on('find_game')
def handle_find_game(data):
    player_id = request.sid
    if request.sid not in active_sessions:
        emit('error', {'message': 'Not authenticated'})
        return

    mode = data.get('mode', 'ranked')
    if mode not in ('ranked', 'casual'):
        mode = 'ranked'

    username = active_sessions[request.sid]
    player_usernames[player_id] = username

    if any(player_id in (g.x_player, g.o_player) for g in games.values()):
        emit('waiting', {'message': 'You are already in a game.'})
        return

    queue = waiting_ranked if mode == 'ranked' else waiting_casual

    if player_id in queue:
        emit('waiting', {'message': 'Still searching for opponent...'})
        return

    opp_id = _find_opponent(queue, username, player_id)

    if opp_id:
        opp_username = player_usernames.get(opp_id, 'Anonymous')
        game_id = str(uuid.uuid4())

        game = TTTGame(game_id, mode=mode)
        game.x_player = opp_id
        game.o_player = player_id
        game.x_username = opp_username
        game.o_username = username
        games[game_id] = game

        join_room(game_id, sid=opp_id)
        join_room(game_id, sid=player_id)

        base = {'game_id': game_id, 'first_turn': 'X', 'new_match': True,
                'x_player': opp_username, 'o_player': username, 'mode': mode}
        emit('game_start', {**base, 'mark': 'X', 'opponent_name': username}, room=opp_id)
        emit('game_start', {**base, 'mark': 'O', 'opponent_name': opp_username}, room=player_id)
    else:
        queue.append(player_id)
        label = 'Ranked' if mode == 'ranked' else 'Casual'
        emit('waiting', {'message': f'Searching for {label} opponent...'})


# ── MOVES ─────────────────────────────────────────────────────────────────────

# SECURITY: Rate limiting to prevent spam attacks
import time

last_move_time = {}
MOVE_COOLDOWN = 0.2  # 200ms minimum between moves per player

# Login rate limiting: max 5 attempts per 30s per socket
login_attempts = {}   # sid -> [timestamp, ...]
LOGIN_MAX = 5
LOGIN_WINDOW = 30


@socketio.on('make_move')
def handle_make_move(data):
    game_id = data.get('game_id')
    index = data.get('index')
    player_id = request.sid

    # ═══════════════════════════════════════════════════════════════
    # SECURITY LAYER 1: Input Validation
    # ═══════════════════════════════════════════════════════════════

    # Validate data types to prevent injection attacks
    if not isinstance(game_id, str) or len(game_id) != 36:
        emit('error', {'message': 'Invalid game ID format'})
        return
    # Validate UUID format (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    import re as _re
    if not _re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', game_id):
        emit('error', {'message': 'Invalid game ID format'})
        return

    if not isinstance(index, int):
        emit('error', {'message': 'Invalid move format'})
        return

    # Validate index bounds (0-8 for tic-tac-toe)
    if index < 0 or index > 8:
        emit('error', {'message': 'Move out of bounds'})
        return

    # ═══════════════════════════════════════════════════════════════
    # SECURITY LAYER 2: Rate Limiting
    # ═══════════════════════════════════════════════════════════════

    current_time = time.time()
    if player_id in last_move_time:
        time_since_last = current_time - last_move_time[player_id]
        if time_since_last < MOVE_COOLDOWN:
            print(f"[SECURITY] Rate limit hit: {player_id}", flush=True)
            emit('error', {'message': 'Too fast! Slow down.'})
            return
    last_move_time[player_id] = current_time

    print(f"[make_move] game={game_id} index={index} player={player_id}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # SECURITY LAYER 3: Game & Player Authentication
    # ═══════════════════════════════════════════════════════════════

    if game_id not in games:
        print(f"[make_move] game not found!", flush=True)
        emit('error', {'message': 'Game not found'})
        return

    try:
        game = games[game_id]

        # Verify player is actually in this game
        if player_id not in [game.x_player, game.o_player]:
            print(f"[SECURITY] Unauthorized player {player_id} tried to move in {game_id}", flush=True)
            emit('error', {'message': 'You are not in this game'})
            return

        mark = game.get_mark(player_id)

        # ═══════════════════════════════════════════════════════════════
        # SECURITY LAYER 4: Server-Side Timer Check
        # ═══════════════════════════════════════════════════════════════

        # Check if turn time expired (server-side validation)
        time_elapsed = current_time - game.turn_start_time
        if time_elapsed > 16:  # 15s + 1s grace period
            print(f"[SECURITY] Move rejected - time expired ({time_elapsed:.1f}s)", flush=True)
            emit('error', {'message': 'Time expired!'})
            # Force timeout
            socketio.start_background_task(handle_timeout, {'game_id': game_id})
            return

        print(f"[make_move] mark={mark} turn={game.turn} board={game.board} time={time_elapsed:.1f}s", flush=True)

        # ═══════════════════════════════════════════════════════════════
        # SECURITY LAYER 5: Server Validates All Game Logic
        # ═══════════════════════════════════════════════════════════════
        # The game.make_move() method validates:
        # - Is it this player's turn?
        # - Is the cell empty?
        # - Is the move legal?

        success, removed_index, game_over, winner, win_line = game.make_move(index, mark)

        print(f"[make_move] success={success} game_over={game_over} winner={winner} win_line={win_line}", flush=True)

        if not success:
            print(f"[SECURITY] Invalid move rejected", flush=True)
            emit('error', {'message': 'Invalid move!'})
            return

        # Reset timer for next turn
        game.turn_start_time = current_time

        stats_x = stats_o = None
        if game_over and winner:
            game.match_score[winner] += 1
            if game.mode == 'ranked':
                stats_x = update_user_stats(
                    game.x_username,
                    'win' if winner == 'X' else 'loss',
                    opponent_username=game.o_username
                )
                stats_o = update_user_stats(
                    game.o_username,
                    'win' if winner == 'O' else 'loss',
                    opponent_username=game.x_username
                )
                print(f"[make_move] ranked stats updated x={stats_x} o={stats_o}", flush=True)
            else:
                print(f"[make_move] casual game — no MMR change", flush=True)

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
        emit('error', {'message': 'An internal server error occurred.'})


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

    new_game = TTTGame(new_game_id, mode=game.mode)
    new_game.x_player = game.x_player
    new_game.o_player = game.o_player
    new_game.x_username = game.x_username
    new_game.o_username = game.o_username
    new_game.match_score = dict(game.match_score)
    new_game.round = game.round
    new_game.turn = first  # SET THE CORRECT STARTING TURN
    games[new_game_id] = new_game

    print(f"[next_round] new game turn={new_game.turn} emitting to X sid={game.x_player}", flush=True)
    _base_n = {'game_id': new_game_id, 'first_turn': first, 'new_match': False,
               'x_player': game.x_username, 'o_player': game.o_username, 'mode': game.mode}
    socketio.emit('rematch_start', {**_base_n, 'mark': 'X', 'opponent_name': game.o_username},
                  room=game.x_player, namespace='/')

    print(f"[next_round] emitting to O sid={game.o_player}", flush=True)
    socketio.emit('rematch_start', {**_base_n, 'mark': 'O', 'opponent_name': game.x_username},
                  room=game.o_player, namespace='/')

    del games[game_id]
    print(f"[next_round] old game deleted", flush=True)


@socketio.on('timeout')
def handle_timeout(data):
    """Player ran out of time — make a random move for them."""
    game_id = data.get('game_id')
    player_id = request.sid

    # Must be authenticated
    if player_id not in active_sessions:
        print(f"[SECURITY] unauthenticated timeout from {player_id}", flush=True)
        return

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
        if game.mode == 'ranked':
            stats_x = update_user_stats(
                game.x_username,
                'win' if winner == 'X' else 'loss',
                opponent_username=game.o_username
            )
            stats_o = update_user_stats(
                game.o_username,
                'win' if winner == 'O' else 'loss',
                opponent_username=game.x_username
            )

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

    # Verify sender is actually in this game
    if player_id not in (game.x_player, game.o_player):
        print(f"[SECURITY] rematch_request rejected: {player_id} not in game {game_id}", flush=True)
        return

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
        new_game = TTTGame(new_game_id, mode=game.mode)
        new_game.x_player = game.x_player
        new_game.o_player = game.o_player
        new_game.x_username = game.x_username
        new_game.o_username = game.o_username
        new_game.match_score = game.match_score
        new_game.round = game.round
        games[new_game_id] = new_game

        join_room(new_game_id, sid=game.x_player)
        join_room(new_game_id, sid=game.o_player)

        _base_r = {'game_id': new_game_id, 'first_turn': first, 'new_match': new_match,
                   'x_player': game.x_username, 'o_player': game.o_username, 'mode': game.mode}
        emit('rematch_start', {**_base_r, 'mark': 'X', 'opponent_name': game.o_username}, room=game.x_player)
        emit('rematch_start', {**_base_r, 'mark': 'O', 'opponent_name': game.x_username}, room=game.o_player)

        del games[game_id]


@socketio.on('leave_game')
def handle_leave_game(data):
    """Player leaves from end screen — cancel any rematch and clean up."""
    game_id = data.get('game_id')
    player_id = request.sid

    print(f"[leave_game] player={player_id} game={game_id}", flush=True)

    if not game_id:
        return

    # Verify sender is actually a player in this game before doing anything destructive
    if game_id in games:
        game = games[game_id]
        if player_id not in (game.x_player, game.o_player):
            print(f"[SECURITY] leave_game rejected: {player_id} not in game {game_id}", flush=True)
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
    player_id = request.sid

    # SECURITY: Validate inputs
    if not isinstance(message, str):
        return

    # SECURITY: Use server-side username — never trust client-supplied sender
    sender = active_sessions.get(player_id, 'Anonymous')

    # SECURITY: Length limits to prevent spam
    if not message or len(message) > 200:
        return

    # SECURITY: Basic XSS prevention - remove HTML tags
    import re
    message = re.sub(r'<[^>]*>', '', message)

    if game_id not in games:
        return

    # SECURITY: Verify player is in this game
    game = games[game_id]
    if player_id not in [game.x_player, game.o_player]:
        return

    emit('chat_message', {'sender': sender, 'message': message}, room=game_id, skip_sid=request.sid)


@socketio.on('get_leaderboard')
def handle_leaderboard():
    if request.sid not in active_sessions:
        return
    emit('leaderboard_data', {'leaderboard': get_leaderboard()})


# ── DISCONNECT ────────────────────────────────────────────────────────────────

@socketio.on('disconnect')
def handle_disconnect():
    player_id = request.sid
    for q in (waiting_casual, waiting_ranked):
        if player_id in q:
            q.remove(player_id)

    # Clean up username -> sid mapping only if this sid is still the active one
    username = active_sessions.get(player_id)
    if username and username_to_sid.get(username) == player_id:
        del username_to_sid[username]

    player_usernames.pop(player_id, None)
    active_sessions.pop(player_id, None)

    last_move_time.pop(player_id, None)   # prevent memory leak
    login_attempts.pop(player_id, None)    # prevent memory leak

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


init_db()

if __name__ == '__main__':
    import os

    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)