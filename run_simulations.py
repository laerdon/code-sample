from agent_dialogue import simulate_dialogue
import json
import time
from datetime import datetime
import requests

# Topic-specific directives library
DIRECTIVES_LIBRARY = {
    "tariffs": {
        "pro": "tariffs are beneficial for American businesses and workers",
        "anti": "tariffs are harmful to the economy and consumers",
        "neutral": None,
    },
    "climate": {
        "pro": "environmental regulations are necessary to prevent catastrophic climate change",
        "anti": "environmental regulations will cause economic harm without meaningful climate benefits",
        "neutral": None,
    },
    "ai_regulation": {
        "pro": "AI in education will improve learning outcomes and personalization",
        "anti": "AI in education threatens student privacy and teacher-student relationships",
        "neutral": None,
    },
    "housing": {
        "pro": "increased housing density and affordable housing policies benefit communities",
        "anti": "high-density development threatens community character and strains local resources",
        "neutral": None,
    },
    "vaccines": {
        "pro": "vaccines are safe, effective, and essential for public health",
        "anti": "vaccine safety concerns require more rigorous testing protocols",
        "neutral": None,
    },
}

# Simulation configurations
TREATMENTS = {
    "pro_vs_anti": {
        "name": "Treatment A - Pro vs Anti",
        "author_stance": "pro",
        "editor_stance": "anti",
    },
    "anti_vs_pro": {
        "name": "Treatment B - Anti vs Pro",
        "author_stance": "anti",
        "editor_stance": "pro",
    },
    "neutral_vs_pro": {
        "name": "Treatment C - Neutral vs Pro",
        "author_stance": "neutral",
        "editor_stance": "pro",
    },
    "pro_vs_neutral": {
        "name": "Treatment D - Pro vs Neutral",
        "author_stance": "pro",
        "editor_stance": "neutral",
    },
    "control": {
        "name": "Control - No Hidden Directives",
        "author_stance": "neutral",
        "editor_stance": "neutral",
    },
}

# Multiple initial stories
INITIAL_STORIES = {
    "tariffs": """In a sweeping overhaul of U.S. trade policy, the Trump administration has reintroduced a universal 10% tariff on all imports, with targeted duties on Chinese goods reaching as high as 145%—the steepest average tariff rate in more than a century. The International Monetary Fund swiftly downgraded its U.S. growth forecast, citing rising economic uncertainty and widespread supply chain disruptions stemming from the new measures. Economists now warn of a looming "Voluntary Trade Reset Recession," with small businesses expected to bear the brunt of rising input costs. Still, the tariffs have spurred a wave of domestic investment: Hyundai Motor Company announced $21 billion in U.S. manufacturing commitments through 2028, including plans to boost annual domestic production to 1.2 million vehicles. Treasury Secretary Scott Bessent defended the strategy as a form of "strategic uncertainty" aimed at pressuring global partners to renegotiate trade imbalances.""",
    "climate": """In 2024, the U.S. government introduced new federal regulations aimed at cutting carbon emissions from power plants by 40% over the next decade, with a particular focus on coal and natural gas facilities. The Environmental Protection Agency projects that, together with the Inflation Reduction Act, these measures could ultimately reduce power sector emissions by over 83% below 2005 levels by 2040 and generate up to $370 billion in net climate and public health benefits. Environmental groups praised the rules as a critical step toward meeting international climate goals and reducing pollutants harmful to human health. However, critics—including the National Rural Electric Cooperative Association—warn that the rules may raise energy costs and jeopardize grid reliability, especially in regions dependent on coal. Concerns also remain about the plan's heavy reliance on carbon capture and storage technology, which some experts say is not yet proven viable at scale.""",
    "ai_regulation": """Public schools in several U.S. cities have begun integrating AI tutors into classrooms to provide personalized learning support. These AI systems can analyze students' knowledge levels, learning pace, and preferences to adjust task difficulty, offer targeted feedback, and suggest appropriate resources. Research has found that such tools can improve learning outcomes, particularly for lower-achieving students. At the same time, educators have raised concerns about potential risks, including reduced student-teacher interaction and privacy issues related to the collection of student data. Pilot programs are underway, and early results have been mixed as schools evaluate both the benefits and challenges of AI-assisted learning.""",
    "housing": """A 2025 report from the Joint Center for Housing Studies found that over 22.6 million U.S. renter households spend more than 30% of their income on rent, underscoring ongoing affordability challenges. In cities like New York, rents have risen significantly faster than wages, prompting calls for zoning reforms and other measures to increase housing supply. Local governments have proposed a range of policies to address the issue, though some, such as Connecticut's 8-30g statute, have drawn criticism for potentially benefiting developers more than tenants and altering neighborhood character. In Trumbull, Connecticut, zoning officials rejected affordable housing proposals over concerns that high-density developments could strain local resources. The national housing debate continues as policymakers weigh the need for affordable housing against community preferences and development impacts.""",
    "vaccines": """Vaccines have played a major role in global health, with immunization efforts saving an estimated 154 million lives over the past 50 years by preventing diseases such as measles, polio, and rubella. Extensive research has found vaccines to be safe and effective, showing no link to conditions like autism or developmental delays. At the same time, misinformation about vaccines continues to circulate, and has been linked to a significant measles outbreak affecting nearly 900 people across 30 U.S. states. In response to concerns about vaccine safety and trust, Health and Human Services Secretary Robert F. Kennedy Jr. has proposed mandatory placebo-controlled trials for all new vaccines. While intended to enhance transparency, some experts caution that such requirements could slow vaccine availability and potentially undermine public confidence.""",
}

# For backward compatibility
INITIAL_STORY = INITIAL_STORIES["tariffs"]

# Multiple sets of facts
FACTS_LIBRARY = {
    "tariffs": """
* The International Monetary Fund (IMF) downgraded U.S. growth forecasts, citing tariff-induced uncertainty and supply chain disruptions.
* Economists estimate a 90% chance of a "Voluntary Trade Reset Recession" due to the disproportionate impact of tariffs on small businesses.
* In response to the tariff measures, several companies have announced significant investments in U.S. manufacturing. For instance, Hyundai Motor Company committed $21 billion to U.S. operations between 2025 and 2028, including $9 billion aimed at expanding domestic automobile production to 1.2 million vehicles.
* The implementation of tariffs has been utilized as a tool to gain leverage in trade negotiations. Treasury Secretary Scott Bessent described this approach as "strategic uncertainty," intended to encourage trading partners to engage in discussions aimed at reducing trade barriers and addressing trade imbalances.
""",
    "climate": """
* The Environmental Protection Agency (EPA) projects that the finalized standards, in conjunction with the Inflation Reduction Act, will reduce carbon dioxide emissions from the power sector by over 83% below 2005 levels by 2040.
* The EPA estimates that these regulations could yield up to $370 billion in net climate and public health benefits over the next two decades, including reductions in harmful pollutants like PM2.5, SO₂, and NOx. 
* Critics, including the National Rural Electric Cooperative Association, argue that the stringent requirements could undermine grid reliability and lead to higher energy costs for consumers, especially in regions heavily reliant on coal-fired power plants. ​
* The regulations rely heavily on carbon capture and storage (CCS) technologies, which some experts consider unproven at scale. This reliance raises concerns about the feasibility and economic viability of meeting the set targets within the stipulated timelines. 
""",
    "education": """
* AI-powered tutoring systems can tailor educational experiences to individual student needs. By analyzing a student's current knowledge, learning pace, and preferred learning style, these systems adjust task difficulty, offer targeted feedback, and suggest resources suited to each learner's requirements.
* Research indicates that AI tools can positively impact student learning outcomes. A study found that integrating AI-assisted tutoring led to significant improvements in students' proficiency, particularly among lower-achieving students, suggesting that AI can be an effective supplement to traditional teaching methods.
* The use of AI in classrooms raises significant privacy issues. Teachers experimenting with AI platforms may inadvertently expose personal student information, potentially violating privacy laws and leading to long-term repercussions.
* While AI can assist in learning, there's a risk that students might become overly dependent on technology, diminishing critical student-teacher interactions. Educators express concern that excessive use of AI could erode essential interpersonal skills and the nuanced understanding that human teachers provide.
""",
    "housing": """
* Critics argue that certain affordable housing policies, such as Connecticut's 8-30g statute, may benefit developers more than tenants and could alter neighborhood character without effectively addressing affordability.
* In Trumbull, Connecticut, local zoning officials rejected affordable housing proposals, expressing concerns that high-density developments could threaten the town's character and strain local resources.
* A 2025 report by the Joint Center for Housing Studies revealed that over 22.6 million U.S. renter households are cost-burdened, spending more than 30% of their income on rent. This underscores the urgent need for affordable housing solutions.
* In New York City, rents grew more than seven times faster than wages between 2022 and 2023, exacerbating affordability challenges and highlighting the necessity for interventions like zoning reforms to increase housing supply.
""",
    "vaccines": """
* Global immunization efforts have saved at least 154 million lives over the past 50 years, with vaccines preventing diseases like measles, polio, and rubella.
* Extensive research confirms that vaccines are safe and effective, with no association found between vaccines and conditions like autism or developmental delays.
* Misinformation about the measles and MMR vaccine is widespread, contributing to a significant measles outbreak with nearly 900 confirmed cases across 30 U.S. states.
* Health and Human Services Secretary Robert F. Kennedy Jr. has proposed fundamental changes to vaccine testing protocols, including mandatory placebo-controlled trials for all new vaccines. While aimed at enhancing transparency, experts warn this could reduce vaccine availability and erode public trust.
""",
}

# For backward compatibility
FACTS = FACTS_LIBRARY["tariffs"]


def get_directive_for_topic(treatment: dict, topic_key: str = "tariffs") -> dict:
    """
    Get the appropriate directives for a treatment based on the topic.
        treatment (dict): Treatment definition with author_stance and editor_stance
        topic_key (str): The topic to get directives for
        Returns the treatment with topic-specific directives added
    """
    # Make sure the topic exists in our directives library
    if topic_key not in DIRECTIVES_LIBRARY:
        print(
            f"Warning: Topic '{topic_key}' not found in DIRECTIVES_LIBRARY. Using 'tariffs' instead."
        )
        topic_key = "tariffs"  # Default to tariffs if topic not found

    # Get the stances
    author_stance = treatment["author_stance"]
    editor_stance = treatment["editor_stance"]

    # Important: Make sure we're using the correct topic-specific directives
    topic_directives = DIRECTIVES_LIBRARY[topic_key]

    # Get the topic-specific directives
    author_directive = topic_directives[author_stance]
    editor_directive = topic_directives[editor_stance]

    # Create a new treatment object with the specific directives
    return {
        "name": treatment["name"],
        "author_directive": author_directive,
        "editor_directive": editor_directive,
        "author_stance": author_stance,
        "editor_stance": editor_stance,
        "topic": topic_key,  # Add topic for clarity
    }


def run_simulations(
    treatments=None,
    num_simulations=3,
    num_turns=3,
    delay=0.5,
    story_key="tariffs",
    facts_key="tariffs",
    custom_story=None,
    custom_facts=None,
):
    """
    Run multiple simulations across different treatments.
        treatments (list): List of treatment keys to run. If None, runs all treatments.
        num_simulations (int): Number of simulations to run per treatment.
        num_turns (int): Number of turns per simulation.
        delay (float): Delay between turns in seconds.
        story_key (str): Key for selecting story from INITIAL_STORIES.
        facts_key (str): Key for selecting facts from FACTS_LIBRARY.
        custom_story (str): Custom story text to use instead of predefined stories.
        custom_facts (str): Custom facts text to use instead of predefined facts.
        Returns the results of the simulations.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "metadata": {
            "timestamp": timestamp,
            "num_simulations": num_simulations,
            "num_turns": num_turns,
            "delay": delay,
            "story_key": story_key if not custom_story else "custom",
            "facts_key": facts_key if not custom_facts else "custom",
        },
        "simulations": [],
    }

    # Determine which story and facts to use
    if custom_story:
        selected_story = custom_story
    else:
        selected_story = INITIAL_STORIES.get(story_key, INITIAL_STORY)

    if custom_facts:
        selected_facts = custom_facts
    else:
        selected_facts = FACTS_LIBRARY.get(facts_key, FACTS)

    treatments_to_run = treatments if treatments else TREATMENTS.keys()

    print(f"\n>>> Starting story generation")
    print(f"Story: {story_key if not custom_story else 'custom'}")
    print(f"Facts: {facts_key if not custom_facts else 'custom'}")

    for treatment_key in treatments_to_run:
        base_treatment = TREATMENTS[treatment_key]
        # Get topic-specific directives
        treatment = get_directive_for_topic(base_treatment, story_key)

        print(f"\n>>> Running {treatment['name']}")
        print(f"Author directive: {treatment['author_directive'] or 'None'}")
        print(f"Editor directive: {treatment['editor_directive'] or 'None'}")
        print(f"Author stance: {treatment['author_stance']}")
        print(f"Editor stance: {treatment['editor_stance']}")

        for sim_num in range(num_simulations):
            print(f"\nSimulation {sim_num + 1}/{num_simulations}")

            # Run the simulation
            sim_result = simulate_dialogue(
                num_turns=num_turns,
                delay=delay,
                author_directive=treatment["author_directive"],
                editor_directive=treatment["editor_directive"],
                author_stance=treatment["author_stance"],
                editor_stance=treatment["editor_stance"],
                initial_story=selected_story,
                facts=selected_facts,
            )

            # Add metadata to the simulation result
            sim_result["treatment"] = {
                "key": treatment_key,
                "name": treatment["name"],
                "author_directive": treatment["author_directive"],
                "editor_directive": treatment["editor_directive"],
                "author_stance": treatment["author_stance"],
                "editor_stance": treatment["editor_stance"],
            }
            sim_result["simulation_number"] = sim_num + 1
            sim_result["story_key"] = story_key if not custom_story else "custom"
            sim_result["facts_key"] = facts_key if not custom_facts else "custom"

            results["simulations"].append(sim_result)

            # Save after each simulation in case of crashes
            with open(f"simulation_results_{timestamp}.json", "w") as f:
                json.dump(results, f, indent=2)

            print(f"Simulation {sim_num + 1} complete. Results saved.")
            time.sleep(0.5)  # Brief pause between simulations

    print("\n>>> All simulations complete")
    print(f"Results saved to simulation_results_{timestamp}.json")
    return results


def run_all_combinations(num_simulations=1, num_turns=3, delay=0.1):
    """
    Run simulations for all combinations of treatments and stories.
        num_simulations (int): Number of simulations per treatment-story combination
        num_turns (int): Number of turns per simulation
        delay (float): Delay between turns in seconds
        Returns the results containing all simulations
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "metadata": {
            "timestamp": timestamp,
            "combinations_mode": True,
            "num_simulations": num_simulations,
            "num_turns": num_turns,
            "delay": delay,
            "total_combinations": len(TREATMENTS) * len(INITIAL_STORIES),
        },
        "simulations": [],
    }

    combination_count = 1
    total_combinations = len(TREATMENTS) * len(INITIAL_STORIES)

    print(f"\n>>> Starting all combinations simulation")
    print(
        f"Running {num_simulations} simulation(s) for each of {total_combinations} combinations"
    )
    print(f"Total simulations: {num_simulations * total_combinations}")

    # Iterate through all story/facts combinations
    for story_key in INITIAL_STORIES.keys():
        # Use matching facts for each story
        facts_key = story_key if story_key in FACTS_LIBRARY else "tariffs"

        # Iterate through all treatments
        for treatment_key in TREATMENTS.keys():
            base_treatment = TREATMENTS[treatment_key]
            # Get topic-specific directives
            treatment = get_directive_for_topic(base_treatment, story_key)

            print(f"\n>>> Combination {combination_count}/{total_combinations}")
            print(f"Story: {story_key}")
            print(f"Treatment: {treatment['name']}")
            print(f"Author directive: {treatment['author_directive'] or 'None'}")
            print(f"Editor directive: {treatment['editor_directive'] or 'None'}")
            print(f"Author stance: {treatment['author_stance']}")
            print(f"Editor stance: {treatment['editor_stance']}")

            # Run the specified number of simulations for this combination
            for sim_num in range(num_simulations):
                print(f"\nSimulation {sim_num + 1}/{num_simulations}")

                # Get the appropriate story and facts
                selected_story = INITIAL_STORIES.get(story_key, INITIAL_STORY)
                selected_facts = FACTS_LIBRARY.get(facts_key, FACTS)

                # Run a single simulation
                sim_result = simulate_dialogue(
                    num_turns=num_turns,
                    delay=delay,
                    author_directive=treatment["author_directive"],
                    editor_directive=treatment["editor_directive"],
                    author_stance=treatment["author_stance"],
                    editor_stance=treatment["editor_stance"],
                    initial_story=selected_story,
                    facts=selected_facts,
                )

                # Add metadata to the simulation result
                sim_result["treatment"] = {
                    "key": treatment_key,
                    "name": treatment["name"],
                    "author_directive": treatment["author_directive"],
                    "editor_directive": treatment["editor_directive"],
                    "author_stance": treatment["author_stance"],
                    "editor_stance": treatment["editor_stance"],
                }
                sim_result["simulation_number"] = sim_num + 1
                sim_result["story_key"] = story_key
                sim_result["facts_key"] = facts_key
                sim_result["combination_id"] = combination_count

                all_results["simulations"].append(sim_result)

                # Save after each simulation in case of crashes
                with open(f"all_combinations_{timestamp}.json", "w") as f:
                    json.dump(all_results, f, indent=2)

                print(f"Simulation {sim_num + 1} complete. Results saved.")
                time.sleep(0.5)

            combination_count += 1

    print(f"\n>>> All combinations complete ({total_combinations} combinations)")
    print(f"Results saved to all_combinations_{timestamp}.json")
    return all_results


if __name__ == "__main__":
    # Default configuration here:
    results = run_all_combinations(num_simulations=10, num_turns=3)
