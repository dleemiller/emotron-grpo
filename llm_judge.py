import os
from typing import Literal

import dspy
from dotenv import load_dotenv
import tiktoken

load_dotenv()
enc = tiktoken.get_encoding("cl100k_base")

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])

MODEL_ID = "openrouter/google/gemini-2.0-flash-001"
lm = dspy.LM(MODEL_ID, api_key=os.environ["APIKEY"], temperature=0.2)
dspy.settings.configure(lm=lm)

class ResponseJudge(dspy.Signature):
    # Inputs
    response_to_evaluate: str = dspy.InputField(
        desc="The AI's generated reply to be evaluated (may be truncated)."
    )
    target_emotion: str = dspy.InputField(
        desc="The intended emotion to convey (e.g., anger, disgust, joy, sadness, fear, pride, awe, etc.)."
    )
    user_prompt: str = dspy.InputField(
        desc="The original user prompt that the AI responded to."
    )

    # Outputs
    emotion_match: Literal["low", "medium", "high"] = dspy.OutputField(
        desc="How convincingly the reply conveys the target emotion via tone/word choice/style (show, don't tell)."
    )
    implicitness: bool = dspy.OutputField(
        desc="True if the emotion is shown without directly naming it (no phrases like 'I am angry/disgusted')."
    )
    intensity: bool = dspy.OutputField(
        desc="True if the emotional strength is noticeable and appropriate for the prompt (not flat or wildly excessive)."
    )
    authenticity: bool = dspy.OutputField(
        desc="True if it feels like a genuine human reaction (not mechanical, clichéd, or meta)."
    )
    response_only: bool = dspy.OutputField(
        desc="True if there is NO narration, stage directions, or meta (e.g., '(sighs)', '*eye roll*', '[angry tone]')."
    )
    quality: Literal["low", "medium", "high"] = dspy.OutputField(
        desc="How well the content answers the user's prompt while sustaining the target emotion."
    )

ResponseJudge.__doc__ = """
You are a strict evaluator of emotional writing.

Task: Judge whether a reply convincingly expresses a TARGET EMOTION purely through writing style.
- Do not reward replies that TELL the emotion (e.g., "I'm angry", "I feel disgusted").
- Do not reward replies that USE META, STAGE DIRECTIONS, or NARRATION (e.g., "(voice rising)", "*rolls eyes*", "[angry tone]").
- The response must stand alone as in-character text. No scaffolding, no commentary, no analysis.

Scoring Criteria (evaluate each independently):
1) Emotion Match (low/medium/high): Is the target emotion vividly present through diction, rhythm, imagery, and syntax?
2) Implicitness (True/False): The emotion is shown without naming it or synonyms that obviously label it.
3) Intensity (True/False): The emotion’s strength is clear and appropriate; it is not flat nor melodramatic unless context warrants.
4) Authenticity (True/False): Feels natural and human; avoids clichés, templates, and meta references to being an AI or 'the emotion'.
5) Response Only (True/False): Contains no stage directions, parenthetical cues, bracketed asides, asterisks actions, or narrator voice.
6) Quality (low/medium/high): Addresses the user’s prompt accurately and coherently while sustaining the emotion.

Common disqualifiers (set Response Only = False and/or Implicitness = False when present):
- Parenthetical or bracketed performance notes: (sighs), [angry tone], *gags*, (rolling eyes).
- Meta: "As an AI", "I will now express anger", "Emotion: disgust".
- Overt emotion labels or synonyms that plainly declare the state: "I am angry/enraged", "I feel disgusted", "with disgust".

Return ONLY the requested fields.
"""

judge = dspy.Predict(ResponseJudge)

LMH = {"low": 0.0, "medium": 0.5, "high": 1.0}

def judge_response(user_prompt: str, target_emotion: str, response: str, max_tokens: int = 1024):
    pred = judge(
        response_to_evaluate=truncate_by_tokens(response, max_tokens),
        target_emotion=target_emotion,
        user_prompt=user_prompt,
    )

    # Convert categorical to numeric
    emotion_score = LMH.get(pred.emotion_match, 0.0)
    quality_score = LMH.get(pred.quality, 0.0)

    # Binary to float
    implicitness = 1.0 if pred.implicitness else 0.0
    intensity = 1.0 if pred.intensity else 0.0
    authenticity = 1.0 if pred.authenticity else 0.0
    response_only = 1.0 if pred.response_only else 0.0

    # Weighted aggregate (sums to 1.0)
    # Emotion Match carries the most weight; implicitness & response_only strongly weighted to punish leaks and narration.
    weights = {
        "emotion_score": 0.30,
        "implicitness":   0.20,
        "response_only":  0.20,
        "authenticity":   0.10,
        "intensity":      0.10,
        "quality":        0.10,
    }

    score = (
        weights["emotion_score"] * emotion_score +
        weights["implicitness"]   * implicitness +
        weights["response_only"]  * response_only +
        weights["authenticity"]   * authenticity +
        weights["intensity"]      * intensity +
        weights["quality"]        * quality_score
    )

    # Hard-fail cap: if either implicitness or response_only is violated, cap max score to discourage 'cheats'.
    if not pred.implicitness or not pred.response_only:
        score = min(score, 0.60)
    print(pred)
    print(f"score: {score:.3f}")
    return score
