import numpy as np

def prompt_generate(agent_position, energy, local_view, local_distances, global_view, global_distances, charging_position):
    # prompt = """As a well-trained resource allocator, you can see local and global observations of the current situation of the agent, including the corresponding future people flow value in the region and the energy required for the agent to reach the region. You need to predict the future position of the agent based on the following scenario:\n"""
    prompt = """
As an experienced game player, you are now required to play a treasure hunt game. The game environment is divided into 6 * 6 squares, each with a treasure chest containing a corresponding amount of coins. You need to control an agent to move to different squares to obtain coins. You need to note that after each time step, the gold coins in the treasure chest will be refreshed, but you can obtain the gold coin information for the next three frames of the treasure chest in advance. And each time you move, it consumes the energy of the agent. You cannot see all the treasure box information in each grid. You can only obtain information about the treasure boxes around the agent, as well as the highest value treasure box information from all the treasure boxes in the eight surrounding areas (top, bottom, left, top, bottom, and right). You need to predict the future position of the agent based on the following scenario:"""
    prompt += "\nLocal observation:\n"
    for direction, value in local_view.items():
        prompt += f"{direction}: Value = {value}, Energy cost = {local_distances[direction][0]}\n"

    # Print global view and distances
    prompt += "\nGlobal observation:\n"
    for direction, value in global_view.items():
        distance = global_distances.get(direction)[0]
        if distance is not None:
            prompt+= f"{direction}: Max value = {value}, Energy cost = {distance}\n"
    
    prompt += f"Charging station position: {charging_position}\n"
    prompt += f"Agent current position: {agent_position}, current energy: {energy}\n"
    # Requirements and hints
    prompt += """
Rules:
- Let's think step by step
- Ensure agent have enough energy to return to the charging station after each selection.
- The agent can choose regions from the local regions, global regions.
- Your goal is to maximize the acquisition of coins under energy constraints. When the energy is sufficient, choose the most valuable treasure chest.
- The chosen position must be identified by the tag: <pos>local/global,direction</pos>
    """
    
    print(prompt)
    return prompt
   