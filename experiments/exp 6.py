# Experiment: Zero-shot, One-shot, and Few-shot Prompting
# Product: Smart Fitness Watch

# Zero-shot Prompt
zero_shot_prompt = """
Write an attractive product description for a Smart Fitness Watch.
Include its key features, health and fitness benefits, battery life,
connectivity, and why customers should buy it.
"""

# One-shot Prompt
one_shot_prompt = """
Example:
Product: Wireless Earbuds
Description: Enjoy clear sound and comfortable listening with these
wireless earbuds. They feature Bluetooth connectivity, a compact
design, long battery life, and a convenient charging case.

Now write a similar product description for a Smart Fitness Watch.
Include its fitness features, health monitoring, connectivity,
battery life, and stylish design.
"""

# Few-shot Prompt
few_shot_prompt = """
Example 1:
Product: Smart Water Bottle
Description: Stay hydrated throughout the day with this smart water
bottle. It reminds you to drink water, tracks your daily intake,
and features a sleek, portable design.

Example 2:
Product: Fitness Earbuds
Description: Make every workout enjoyable with these wireless fitness
earbuds. They offer clear audio, Bluetooth connectivity, sweat
resistance, comfortable fitting, and long battery life.

Example 3:
Product: Smart Scale
Description: Monitor your fitness progress with this smart scale.
It measures body weight and provides useful body composition data
through a connected mobile application.

Task:
Using the style and structure of the examples above, write a product
description for a Smart Fitness Watch. Mention its health monitoring,
fitness tracking, smart notifications, connectivity, battery life,
and comfortable design.
"""

# Display the prompts
print("=" * 60)
print("ZERO-SHOT PROMPT")
print("=" * 60)
print(zero_shot_prompt)

print("=" * 60)
print("ONE-SHOT PROMPT")
print("=" * 60)
print(one_shot_prompt)

print("=" * 60)
print("FEW-SHOT PROMPT")
print("=" * 60)
print(few_shot_prompt)
