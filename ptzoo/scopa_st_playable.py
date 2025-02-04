import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import streamlit as st

# Import your environment, logger, and SB3 libraries.
from maenv.ma_scopa_env import MaScopaEnv
from tlogger import TLogger
from sb3_contrib import MaskablePPO

###############################################
# 1. Preload Card Images
###############################################
def preload_card_images(
    image_folder='C:/Users/aless/Repos/Rug/P2/ScopAI/ScopaAI_ToM/res/cards',
    scale_factor=0.1
):
    """
    Load and scale card images.
    Returns a dictionary mapping card indices to scaled PIL images.
    """
    card_images = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            card_name = filename.split(".")[0]  # e.g., "ace_of_hearts"
            rank, suit = card_name.split("_of_")
            suit_values = {"diamonds": 30, "clubs": 20, "spades": 10, "hearts": 0}
            suit_value = suit_values[suit]
            if rank == "jack":
                rank = 8
            elif rank == "queen":
                rank = 9
            elif rank == "king":
                rank = 10
            elif rank == "ace":
                rank = 1
            card_index = int(rank) + suit_value - 1
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)
            new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
            scaled_image = image.resize(new_size)
            card_images[card_index] = scaled_image
    return card_images

###############################################
# 2. Plot the Game State
###############################################
def plot_game_state(observation, card_images, agent, highlight_action=None, is_current_turn=False):
    """
    Plot a 6x40 grid representing the game state using card images.
    • Only the first three rows (Player's Hand, Table, Player's Capture) are used.
    • If highlight_action is provided, the corresponding cell in row 0 is outlined in red.
    • If is_current_turn is True, the board title and spines are highlighted in red.
    """
    # We'll display only the first 3 rows.
    observation = observation[:3]
    n_rows, n_cols = observation.shape
    row_names = ["Player's Hand", "Table", "Player's Capture"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw grid cells.
    for i in range(n_rows):
        for j in range(n_cols):
            rect = plt.Rectangle((j, i), 1, 1,
                                 fill=False,
                                 edgecolor='black',
                                 linestyle='dotted',
                                 linewidth=0.5)
            ax.add_patch(rect)
    
    # Place card images.
    for i in range(n_rows):
        for j in range(n_cols):
            if observation[i, j] == 1:
                if j in card_images:
                    img_array = np.array(card_images[j])
                    im = OffsetImage(img_array, zoom=1.5)
                    ab = AnnotationBbox(im, (j + 0.5, i + 0.5), frameon=False)
                    ax.add_artist(ab)
    
    # Optionally highlight a card in the player's hand (row 0).
    if highlight_action is not None:
        highlight = plt.Rectangle((highlight_action, 0), 1, 1,
                                  linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(highlight)
    
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(row_names)
    ax.invert_yaxis()
    
    # If this board belongs to the agent whose turn it is, change the title and spines.
    if is_current_turn:
        ax.set_title(f"{agent} (Current Turn)", fontsize=16, color='red')
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
    else:
        ax.set_title(agent, fontsize=14)
    
    plt.tight_layout()
    return fig

###############################################
# 3. List & Load AI Model for Non-Human Players
###############################################
def list_models(model_folder):
    """List all .zip model files in the folder."""
    models = [f for f in os.listdir(model_folder) if f.endswith(".zip")]
    model_dict = {m: os.path.join(model_folder, m) for m in models}
    return models, model_dict

@st.cache_resource
def load_model(model_path):
    return MaskablePPO.load(model_path)

###############################################
# 4. Interactive Game Setup and Loop
###############################################
st.title("Interactive Scopa Game: Play Against the AI")

# Sidebar: Game settings.
st.sidebar.header("Game Setup")
human_agent = st.sidebar.selectbox("Select your player", options=["player_0", "player_1", "player_2", "player_3"])

model_folder = st.sidebar.text_input("Model Folder", value="C:/Users/aless/Repos/Rug/P2/ScopAI/ScopaAI_ToM")
if os.path.isdir(model_folder):
    model_names, model_dict = list_models(model_folder)
else:
    model_names, model_dict = [], {}
    
selected_model_name = st.sidebar.selectbox("Select AI Model for Other Players", options=["None"] + model_names)
if selected_model_name != "None":
    ai_model = load_model(model_dict[selected_model_name])
else:
    ai_model = None

# Sidebar: Auto-play AI switch.
auto_ai = st.sidebar.checkbox("Auto-play AI", value=False)

# "Start New Game" button: Generate a NEW random game.
if st.sidebar.button("Start New Game"):
    tlogger = TLogger("runs/interactive_game")
    env = MaScopaEnv(tlogger=tlogger, render_mode=None)
    # Generate a new random seed each time.
    seed_val = random.randint(1, 10000)
    env.reset(seed=seed_val)
    st.session_state.env = env
    st.session_state.human_agent = human_agent
    st.session_state.done = False
    st.session_state.current_agent = env.agent_selection
    st.session_state.last_action = None  # For logging the last move.

# Preload card images.
card_images = preload_card_images(
    image_folder='C:/Users/aless/Repos/Rug/P2/ScopAI/ScopaAI_ToM/res/cards',
    scale_factor=0.1
)

def index_to_card(index):
    if not 0 <= index < 40:
        raise ValueError("Index must be between 0 and 39.")
    suits = ['di cuori', 'di picche', 'di fiori', 'bello']
    suit_index = index // 10
    suit = suits[suit_index]
    rank = (index % 10) + 1
    if rank == 8:
        rank = 'Jack'
    elif rank == 9:
        rank = 'Donna'
    elif rank == 10:
        rank = 'Re'
    elif rank == 1:
        rank = 'Asso'
    return f'{rank} {suit}'

# Main game loop.
if "env" in st.session_state:
    env = st.session_state.env
    current_agent = env.agent_selection
    st.write(f"**Current Turn:** {current_agent}")
    
    # Display state of all players.
    st.subheader("Game State")
    cols = st.columns(4)
    for idx, agent in enumerate(env.possible_agents):
        try:
            obs = env.observe(agent)
        except Exception:
            if agent == current_agent:
                obs, _, _, _, _ = env.last()
            else:
                obs = np.zeros((6, 40), dtype=int)
        is_current = (agent == current_agent)
        # No celebration parameter here.
        fig = plot_game_state(obs, card_images, agent, highlight_action=None, is_current_turn=is_current)
        cols[idx].pyplot(fig)
    
    # Get current agent's info.
    obs, reward, termination, truncation, info = env.last()
    action_mask = info.get('action_mask', None)
    
    if termination or truncation:
        st.write("**Game Over!**")
        st.session_state.done = True
    else:
        # Human turn.
        if current_agent == st.session_state.human_agent:
            st.write("**Your Turn!**")
            if ai_model is not None:
                best_estimate = ai_model.predict(obs, action_masks=action_mask)[0]
                st.write(f"AI Model {selected_model_name} estimates best card: **{index_to_card(best_estimate)}**")
            if action_mask is not None:
                valid_actions = [f'{i}\t|\t{index_to_card(i)}' for i, valid in enumerate(action_mask) if valid == 1]
            else:
                valid_actions = []
            chosen_action = st.radio("Select your action", options=valid_actions).split("|")[0]
            if st.button("Make Move"):
                env.step(int(chosen_action))
                st.session_state.current_agent = env.agent_selection
                st.session_state.last_action = int(chosen_action)
                st.rerun()
        else:
            st.write(f"**{current_agent}'s Turn (AI)**")
            if action_mask is not None:
                valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
            else:
                valid_actions = []
            if ai_model is not None and valid_actions:
                try:
                    predicted_action = int(ai_model.predict(obs, action_masks=action_mask)[0])
                except Exception as e:
                    st.error(f"AI prediction error: {e} | Random action chosen.")
                    predicted_action = random.choice(valid_actions) if valid_actions else None
            else:
                predicted_action = random.choice(valid_actions) if valid_actions else None
            st.write(f"AI chooses action: **{index_to_card(predicted_action)}**")
            if auto_ai:
                env.step(predicted_action)
                st.session_state.current_agent = env.agent_selection
                st.session_state.last_action = predicted_action
                st.rerun()
            else:
                if st.button("Next Move for AI"):
                    env.step(predicted_action)
                    st.session_state.current_agent = env.agent_selection
                    st.session_state.last_action = predicted_action
                    st.rerun()
else:
    st.write("Click **Start New Game** in the sidebar to begin.")
