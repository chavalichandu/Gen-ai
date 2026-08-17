# Lab 10: Strategy Evaluation

def zero_shot():
    return """
Classify the sentiment of the sentence:
'I really enjoyed this movie.'

Answer:
"""

def one_shot():
    return """
Example:

Sentence: 'The food was terrible.'

Answer: Negative

Now classify:

Sentence: 'I really enjoyed this movie.'

Answer:
"""

def few_shot():
    return """
Example 1

Sentence: 'The food was terrible.'

Answer: Negative

Example 2

Sentence: 'The book is excellent.'

Answer: Positive

Example 3

Sentence: 'The service was disappointing.'

Answer: Negative

Now classify:

Sentence: 'I really enjoyed this movie.'

Answer:
"""

def structured():
    return """
Task:
Sentiment Analysis

Sentence:
'I really enjoyed this movie.'

Return the answer in the following format:

Sentiment: <Positive/Negative>
Confidence: <Percentage>
Reason: <Short Explanation>
"""

print("========== ZERO-SHOT ==========")
print(zero_shot())

print("\n========== ONE-SHOT ==========")
print(one_shot())

print("\n========== FEW-SHOT ==========")
print(few_shot())

print("\n========== STRUCTURED PROMPT ==========")
print(structured())