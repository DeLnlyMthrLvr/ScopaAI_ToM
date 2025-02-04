import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import streamlit as st

# Import your environment, logger, and SB3 libraries
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
    Preload and scale card images.
    
    Args:
      image_folder (str): Path to the folder containing card images.
      scale_factor (float): Scaling factor.
      
    Returns:
      dict: Mapping from card index to a scaled PIL image.
    """
    card_images = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            # Expect filenames like "rank_of_suit.png"
            card_name = filename.split(".")[0]
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
# 2. Plot the Game State (One Pyplot per State)
###############################################
def plot_game_state(observation, card_images, agent, selected_card=None):
    """
    Create a Matplotlib figure of the game state using card images.
    
    Parameters:
      observation (np.array): (6, 40) array representing the zones.
      card_images (dict): Mapping from card index to PIL image.
      agent (str): Agent name.
      selected_card (int or None): If provided, the corresponding cell in row 0 will be highlighted.
                                   (Not used here if you want only one pyplot.)
      
    Returns:
      fig: The Matplotlib figure.
    """
    observation = observation[:3]
    n_rows, n_cols = observation.shape
    row_names = [
        "Player's Hand", "Table", "Player's Capture"
    ]
    
    # Create a larger figure.
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw a dotted grid.
    for i in range(n_rows):
        for j in range(n_cols):
            cell = plt.Rectangle((j, i), 1, 1,
                                 fill=False,
                                 edgecolor='black',
                                 linestyle='dotted',
                                 linewidth=0.5)
            ax.add_patch(cell)
    
    # Stamp card images on cells where observation==1.
    for i in range(n_rows):
        for j in range(n_cols):
            if observation[i, j] == 1:
                if j in card_images:
                    img_array = np.array(card_images[j])
                    im = OffsetImage(img_array, zoom=1.5)
                    ab = AnnotationBbox(im, (j + 0.5, i + 0.5), frameon=False)
                    ax.add_artist(ab)
                else:
                    print(f"No image found for card index {j}")
    
    # (If desired, you could highlight a predicted card here, but for a single state plot we leave it unhighlighted.)
    if selected_card is not None:
        # For instance, if you wanted to highlight one prediction:
        highlight = plt.Rectangle((selected_card, 0), 1, 1,
                                  linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(highlight)
    
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(row_names)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

###############################################
# 3. Simulate a Full Game (All Rounds)
###############################################
def simulate_game_full(episode, tlogger):
    """
    Simulate an entire game (all rounds) using MaScopaEnv.
    A "round" is defined as one full cycle of agent moves.
    
    Args:
      episode (int): Seed for simulation.
      tlogger: Logger instance.
      
    Returns:
      game_rounds: List of rounds. Each round is a dict mapping agent names
                   to a tuple (observation, action_mask).
    """
    env = MaScopaEnv(tlogger=tlogger, render_mode=None)
    env.reset(seed=episode)
    game_rounds = []
    current_round = {}
    first_agent = env.agent_selection

    while True:
        agent = env.agent_selection
        obs, reward, termination, truncation, info = env.last()
        current_round[agent] = (obs, info['action_mask'])
        if termination or truncation:
            env.step(None)
        else:
            action_mask = info['action_mask']
            possible_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
            action = random.choice(possible_actions) if possible_actions else None
            env.step(action)
        
        # When the agent cycle wraps around, record the round.
        if env.agent_selection == first_agent:
            game_rounds.append(current_round)
            current_round = {}
        
        if termination or truncation:
            if current_round:
                game_rounds.append(current_round)
            break

    env.close()
    return game_rounds

###############################################
# 4. List Available Models from a Folder
###############################################
def list_models(model_folder):
    """
    List all .zip model files in the folder.
    
    Returns:
      model_names: List of filenames.
      model_dict: Mapping from filename to full path.
    """
    model_paths = glob.glob(os.path.join(model_folder, "*.zip"))
    model_names = [os.path.basename(path) for path in model_paths]
    model_dict = {os.path.basename(path): path for path in model_paths}
    return model_names, model_dict

# (Optional) Cache the loaded models to avoid reloading on every run.
@st.cache_resource
def load_models(selected_model_names, model_dict):
    models = {}
    for model_name in selected_model_names:
        model_path = model_dict.get(model_name)
        if model_path:
            models[model_name] = MaskablePPO.load(model_path)
    return models

###############################################
# 5. Streamlit Interface
###############################################
st.title("Card Game: Evaluate Selected Models Across Rounds")

# Sidebar: Global episode slider and simulation button.
episode = st.sidebar.slider("Select Episode (Seed)", min_value=1, max_value=10, value=1)
simulate_button = st.sidebar.button("Simulate Full Game")

# Sidebar: Model folder input.
model_folder = st.sidebar.text_input("Model Folder", value="C:/Users/aless/Repos/Rug/P2/ScopAI/ScopaAI_ToM")
model_names, model_dict = ([], {})
if os.path.isdir(model_folder):
    model_names, model_dict = list_models(model_folder)
else:
    st.sidebar.warning("Specified model folder does not exist.")

if not model_names:
    st.sidebar.info("No model files found in the folder.")

# Sidebar: Multi-select widget to choose which models to evaluate.
selected_model_names = st.sidebar.multiselect("Select Models to Evaluate", options=model_names, default=model_names)

# Load the selected models (if any) via caching.
models = {}
if selected_model_names:
    models = load_models(selected_model_names, model_dict)

# When the "Simulate Full Game" button is pressed, run the simulation.
if simulate_button:
    experiment_name = "Simulation_Game"
    tlogger = TLogger(f"runs/{experiment_name}")
    game_rounds = simulate_game_full(episode, tlogger)  # list of rounds (each round is a dict)
    st.session_state["game_rounds"] = game_rounds
    st.session_state["episode"] = episode

# Preload card images.
card_images = preload_card_images(
    image_folder='C:/Users/aless/Repos/Rug/P2/ScopAI/ScopaAI_ToM/res/cards',
    scale_factor=0.1
)

def index_to_card(index):
    if not 0 <= index < 40:
        raise ValueError("Index must be between 0 and 39.")

    # Determine the suit using integer division by 10.
    suits = ['di cuori', 'di picche', 'di fiori', 'bello']
    suit_index = index // 10
    suit = suits[suit_index]

    # Determine the rank using modulo operation.
    rank = (index % 10) + 1

    # Special cases for rank 8, 9, 10, 1.
    if rank == 8:
        rank = 'Jack'
    elif rank == 9:
        rank = 'Donna'
    elif rank == 10:
        rank = 'Re'
    elif rank == 1:
        rank = 'Asso'

    return f'{rank} {suit}'



# If simulation data exists, display every round.
if "game_rounds" in st.session_state:
    st.write(f"### Episode: {st.session_state['episode']}")
    game_rounds = st.session_state["game_rounds"]
    
    # For each round...
    for r_idx, round_state in enumerate(game_rounds, start=1):
        # For each agent in the round, display one state plot and then an expander for predictions.
        cols = st.columns(4)
        st.write(f"#### Round |{r_idx}|")
        for i, agent in enumerate(round_state.keys()):
            with cols[i]:
                st.markdown(f"**Agent: {agent}**")
                obs, action_mask = round_state[agent]
                # Display the state once (without highlighting any predicted card)
                fig = plot_game_state(obs, card_images, agent, selected_card=None)
                st.pyplot(fig)
                # In an expander, list the predictions from each selected model.
                
                for model_name in selected_model_names:
                    model = models.get(model_name)
                    if model is not None:
                        try:
                            if 'ToM' not in model_name:
                                predicted_action = int(model.predict(obs[:3], action_masks=action_mask)[0])
                            else:
                                predicted_action = int(model.predict(obs, action_masks=action_mask)[0])
                        except Exception as e:
                            predicted_action = None
                            st.error(f"Error in {model_name}: {e}")

                        if model_name == 'scopa_v0_ToM1_20250125-205659.zip':
                            st.write(f"👑**{model_name}:** {index_to_card(predicted_action)}")
                        else:
                            st.write(f"**{model_name}:** {index_to_card(predicted_action)}")
else:
    st.write("Click **Simulate Full Game** in the sidebar to simulate a game and view predictions.")
