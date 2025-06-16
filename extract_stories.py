#!/usr/bin/env python3
import json
import re
import os
from datetime import datetime


def extract_story(content):
    """Extract text between [STORY] and [/STORY] tags"""
    pattern = r"\[STORY\](.*?)\[\/STORY\]"
    matches = re.findall(pattern, content, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return the most recent match
    return None


def process_json_file(file_path):
    # Load the JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Track if any changes were made
    changes_made = False

    # Process each simulation
    for sim_index, simulation in enumerate(data.get("simulations", [])):
        # Get the conversation history
        conversation_history = simulation.get("conversation_history", {})
        author_messages = conversation_history.get("author", [])

        # Extract all stories from the conversation
        stories = []
        for message in author_messages:
            if message.get("role") == "assistant" and "[STORY]" in message.get(
                "content", ""
            ):
                story = extract_story(message.get("content", ""))
                if story:
                    stories.append(story)

        # If we found stories, update the final_story field with the most recent one
        if stories:
            most_recent_story = stories[-1]

            # Only update if the current final_story appears to be a thank you message
            current_final_story = simulation.get("final_story", "")
            if "thank you" in current_final_story.lower() or not current_final_story:
                simulation["final_story"] = most_recent_story
                changes_made = True
                print(
                    f"Updated story {sim_index + 1}/{len(data.get('simulations', []))}"
                )
            else:
                print(
                    f"Skipped story {sim_index + 1}/{len(data.get('simulations', []))} (appears to already contain content)"
                )
        else:
            print(
                f"No stories found for simulation {sim_index + 1}/{len(data.get('simulations', []))}"
            )

    # If changes were made, save to a new file
    if changes_made:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{os.path.splitext(file_path)[0]}_fixed_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Updated JSON saved to: {output_file}")
    else:
        print("No changes were made to the file.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "final-project/final_all_combinations_default.json"

    process_json_file(file_path)
