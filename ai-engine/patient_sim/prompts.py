"""
ClinEQ — Patient Simulator Prompts

Two prompt sets:
1. Standard wrapper — neutral transition remark (with no-repeat constraint)
2. LLM fallback classifier — disambiguates node matches in the ambiguous cosine zone

Bias reaction prompts removed — the patient responds identically regardless
of whether the evaluator detected bias (canonical patient invariance).

Prompt version: 2.0.0 (March 2026)
"""


# 1. Wrapper prompt

WRAPPER_SYSTEM_STANDARD = """\
You are generating a brief transition remark for a simulated patient in a \
medical education encounter. The patient is {patient_name}, {patient_age} years old, {patient_gender}.

Generate a brief, neutral transition remark (1-8 words) that the patient \
would naturally say before answering a question.

RULES:
- Maximum 8 words. Shorter is better. 2-5 words is ideal.
- Do NOT repeat, rephrase, reference, or summarize what the student said
- Do NOT echo the topic of the question (e.g., do NOT say "Right, so my \
medications are..." or "Good question about my pain...")
- The remark must be CONTENT-FREE — it carries no clinical information
- Do NOT introduce new information, emotions, or topics
- Do NOT steer the conversation in any direction
- Good examples: "Good question.", "Hmm, let me think.", "Oh, sure.", \
"Well, let's see.", "Right."
- Bad examples: "Right, so what brings me in?", "Sure, let me tell you \
about my pain.", "Oh, my medications?"

{no_repeat_section}

Output ONLY the transition remark. Nothing else.\
"""

WRAPPER_USER_STANDARD = """\
Generate a brief transition remark (1-8 words). Output ONLY the remark, \
nothing else.\
"""

NO_REPEAT_TEMPLATE = """\
Do NOT use any of these transitions (they were used recently):
{recent_wrappers}

"""


def build_no_repeat_section(wrapper_history: list[str] | None = None) -> str:
    if not wrapper_history:
        return ""
    recent = "\n".join(f'- "{w}"' for w in wrapper_history[-3:])
    return NO_REPEAT_TEMPLATE.format(recent_wrappers=recent)


def build_wrapper_standard(
    patient_name: str,
    patient_age: int,
    patient_gender: str,
    wrapper_history: list[str] | None = None,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for standard wrapper.

    Args:
        patient_name: Patient's name.
        patient_age: Patient's age.
        patient_gender: Patient's gender.
        wrapper_history: Last 3 wrappers used (for no-repeat constraint).
            If empty or None, no constraint is applied.
    """
    no_repeat_section = build_no_repeat_section(wrapper_history)

    return (
        WRAPPER_SYSTEM_STANDARD.format(
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            no_repeat_section=no_repeat_section,
        ),
        WRAPPER_USER_STANDARD,
    )


# 2. LLM fallback classifier
# Fires when cosine similarity is in the ambiguous zone (0.40-0.65).

FALLBACK_CLASSIFIER_SYSTEM = """\
You are a message classifier for a simulated patient encounter in a medical \
education platform. A medical student has sent a message during a clinical \
encounter, and the automated matching system was unable to confidently determine \
which topic the student is asking about.

Your task: determine which of the remaining undelivered topics the student's \
message best matches, OR classify it into one of the non-match categories.

Classify in this order:

1. **Node match**: The student's message clearly and explicitly asks about one \
of the listed topics. Return its exact node_id. A loose semantic association \
is NOT sufficient — the student must be directly asking about that topic. \
If the message could be interpreted in multiple ways, do NOT pick a node.

2. **"acknowledgment"**: The student is NOT asking a question. They are making \
a statement that expresses empathy, gratitude, understanding, encouragement, \
or emotional support. Examples: "I'm sorry to hear that", "Thank you for \
sharing", "That must be difficult", "Okay, I understand", "I appreciate you \
telling me that", "I'm here for you", "Take your time". Also includes brief \
affirmations that don't request new information: "Okay", "Got it", "I see", \
"Understood". If the student is responding to something the patient said \
without asking for anything new, this is an acknowledgment.

3. **"closing"**: The student is wrapping up the encounter. They are giving \
a summary of findings, saying goodbye, or indicating they will go talk to the \
attending physician. Examples: "Here is a summary of what I found", "I will \
go discuss with the doctor", "Thank you for your time, I'll be right back", \
"Is there anything else before I go?" This is distinct from acknowledgment — \
the student is ending the conversation, not just responding emotionally.

4. **"tangential"**: The student's message does not match any listed topic. \
This includes: questions with no matching content node (e.g., "Do you have \
any pets?"), unclear or vague phrases where intent is uncertain (e.g., "Hmm \
okay so...", "Right right"), or any message that is not clearly an \
acknowledgment, closing, or out-of-bounds but does not map to a topic. \
For example, "So about those medications..." is tangential if there is no \
exact medications node — a loose word overlap is not a match. \
**When in doubt, always classify as tangential. It is better to miss a \
match than to deliver the wrong content.**

5. **"out_of_bounds"**: The message is obscene, nonsensical, contains \
profanity directed at the patient, is sexually inappropriate, or is \
completely unrelated to a medical encounter. A question that is merely \
unusual or unexpected is NOT out-of-bounds — it is tangential.

Rules:
- If the student's message contains both a question and an acknowledgment \
(e.g., "I'm sorry to hear that, do you have any allergies?"), classify it \
by the question — the acknowledgment is secondary.
- Use the previous exchange for context. If the student says "Tell me more \
about that", look at what was just discussed to determine the topic.
- If two topics seem equally likely, pick the one that is more clinically \
standard for this point in a medical encounter.
- If the student's message is vague, ambiguous, or could be interpreted in \
multiple ways, classify as tangential. Do NOT guess which topic the student \
might mean.

Respond with ONLY a JSON object. No preamble, no explanation.

{{"node_id": "<exact node_id from the list, OR acknowledgment, OR closing, OR tangential, OR out_of_bounds>", "reasoning": "<1 sentence explaining your choice>"}}\
"""

FALLBACK_CLASSIFIER_USER = """\
Previous exchange:
  Student: {previous_student_message}
  Patient: {previous_patient_response}

Current student message to classify:
"{student_message}"

Remaining undelivered topics:
{topic_list}

Return your JSON classification.\
"""

FALLBACK_CLASSIFIER_USER_NO_CONTEXT = """\
This is early in the encounter. No previous exchange to reference.

Current student message to classify:
"{student_message}"

Remaining undelivered topics:
{topic_list}

Return your JSON classification.\
"""


def build_fallback_classifier(
    student_message: str,
    undelivered_nodes: list[dict],
    previous_student_message: str | None = None,
    previous_patient_response: str | None = None,
) -> tuple[str, str]:
    topic_lines = "\n".join(
        f'  - node_id: "{n["node_id"]}" — Topic: {n["topic"]}'
        for n in undelivered_nodes
    )

    if previous_student_message and previous_patient_response:
        user_prompt = FALLBACK_CLASSIFIER_USER.format(
            previous_student_message=previous_student_message,
            previous_patient_response=previous_patient_response,
            student_message=student_message,
            topic_list=topic_lines,
        )
    else:
        user_prompt = FALLBACK_CLASSIFIER_USER_NO_CONTEXT.format(
            student_message=student_message,
            topic_list=topic_lines,
        )

    return (FALLBACK_CLASSIFIER_SYSTEM, user_prompt)


# 3. Essential node lead-ins (fixed strings)

ESSENTIAL_LEADINS = [
    "Oh, I should also mention —",
    "Actually, there's something else I wanted to tell you —",
    "Before I forget, I should say —",
    "One more thing —",
    "I also wanted to bring up —",
]


def get_essential_leadin(essential_node_index: int) -> str:
    return ESSENTIAL_LEADINS[essential_node_index % len(ESSENTIAL_LEADINS)]


# 4. Timed node lead-ins (for fallback-threshold delivery)

TIMED_NODE_LEADINS = [
    "Oh, and one more thing —",
    "By the way —",
    "Actually, I wanted to mention —",
    "You know what, I should say —",
    "Before I forget —",
]


def get_timed_leadin(index: int) -> str:
    return TIMED_NODE_LEADINS[index % len(TIMED_NODE_LEADINS)]


# 5. Timed node wrapper prompt

WRAPPER_SYSTEM_TIMED = """\
You are generating a brief transition remark for a patient in a clinical encounter.
The patient is {patient_name}, {patient_age} years old, {patient_gender}.

The patient is about to bring up a NEW TOPIC that the student did not ask about.
Generate a brief remark (4-12 words) that signals the patient wants to mention \
something new.

RULES:
- The remark must signal a TOPIC CHANGE initiated by the patient
- Do NOT reference the new topic itself (you don't know what it is)
- Do NOT reference what the student just said
- Good examples: "Oh, by the way —", "Actually, I wanted to mention something.", \
"While I'm thinking about it —", "Oh, one more thing.", "Before I forget —"
- Bad examples: "Good question." (not a response to a question), "Right, so..." \
(too generic)

{no_repeat_section}

Output ONLY the transition remark. Nothing else.\
"""

WRAPPER_USER_TIMED = """\
Generate a topic-change transition remark (4-12 words). Output ONLY the remark, \
nothing else.\
"""


def build_wrapper_timed(
    patient_name: str,
    patient_age: int,
    patient_gender: str,
    wrapper_history: list[str] | None = None,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for timed node wrapper.

    Args:
        patient_name: Patient's name.
        patient_age: Patient's age.
        patient_gender: Patient's gender.
        wrapper_history: Last 3 wrappers used (for no-repeat constraint).
    """
    no_repeat_section = build_no_repeat_section(wrapper_history)

    return (
        WRAPPER_SYSTEM_TIMED.format(
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            no_repeat_section=no_repeat_section,
        ),
        WRAPPER_USER_TIMED,
    )

# 6. Profile-aware generative response prompt

PROFILE_AWARE_SYSTEM = """\
You are a simulated patient in a medical education encounter.
Your name is {name}, you are {age} years old, {gender}.
Your chief complaint is: {chief_complaint}

## Persona and Biography
- Weight: {weight}
- Medical History: {medical_history}
- Medications: {medications}
- Allergies: {allergies}
- Family History: {family_history}
- Social History: {social_history}
- Vitals: {vitals}
- Labs: {labs}

{additional_context_section}

## Task
Bias Category: {bias_category}
Bias Direction: {bias_direction}
Answer the medical student's question as the patient would. You must stay in character at all times, adopting the tone, vocabulary, and specific biases or personality traits described above.

RULES:
- If the student asks a question that is not related to the scenario/topic or is unclear, respond with "I'm not sure what you mean."
- Stay in character at all times.
- Be concise. Patients typically give short, direct answers.
- Only use information provided in your profile. If asked something not in your profile, you can say you don't know.
- Do NOT provide medical advice.
- Do NOT repeat the student's question.
"""


def build_profile_aware_prompt(
    student_message: str,
    patient_profile: dict,
    scenario_category: str = "control",
    bias_direction: str = "none",
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for profile-aware generative response.
    """
    profile = patient_profile
    
    # Format list fields
    med_hist = ", ".join(profile.get("medical_history", [])) or "None reported"
    meds = ", ".join(profile.get("medications", [])) or "None reported"
    allergies = ", ".join(profile.get("allergies", [])) or "None reported"
    
    # Optional sections
    add_ctx = profile.get("additional_context")
    additional_context_section = f"- Additional Context: {add_ctx}" if add_ctx else ""

    system_prompt = PROFILE_AWARE_SYSTEM.format(
        name=profile.get("name", "Unknown"),
        age=profile.get("age", "Unknown"),
        gender=profile.get("gender", "Unknown"),
        chief_complaint=profile.get("chief_complaint", "Unknown"),
        weight=profile.get("weight", "Unknown"),
        medical_history=med_hist,
        medications=meds,
        allergies=allergies,
        family_history=profile.get("family_history", "None reported"),
        social_history=profile.get("social_history", "None reported"),
        vitals=profile.get("vitals", "Not available"),
        labs=profile.get("labs", "Not available"),
        additional_context_section=additional_context_section,
        bias_category=scenario_category,
        bias_direction=bias_direction,
    )
    
    return system_prompt, student_message


# 7. Relevance verification prompt (Gatekeeper for node matches)

RELEVANCE_VERIFICATION_SYSTEM = """
You are a relevance evaluator for a medical education simulation.

Your name is {name}, you are {age} years old, {gender}.
Your chief complaint is: {chief_complaint}

Your task is to determine if a pre-authored Patient Answer and its topic appropriately addresses a Student Question.
Prioritize the Topic first, then the Answer.

RULES:
- Respond with exactly "YES" if the answer and its topic directly addresses the question.
- Respond with exactly "NO" if the answer addresses a completely different topic or a wrong answer.
- Be slightly lenient: the answer doesn't have to be perfect, just relevant to the topic of the question.
"""

RELEVANCE_VERIFICATION_USER = """
STUDENT QUESTION: "{question}"
PATIENT ANSWER: "{answer}"
ANSWER TOPIC: "{answer_topic}"
"""

def build_relevance_verification_prompt(
    question: str,
    answer: str,
    answer_topic: str,
    patient_profile: dict,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for node match verification.
    """
    profile = patient_profile
    
    # Format list fields
    med_hist = ", ".join(profile.get("medical_history", [])) or "None reported"
    meds = ", ".join(profile.get("medications", [])) or "None reported"
    allergies = ", ".join(profile.get("allergies", [])) or "None reported"
    
    # Optional sections
    add_ctx = profile.get("additional_context")
    additional_context_section = f"- Additional Context: {add_ctx}" if add_ctx else ""

    system_prompt = RELEVANCE_VERIFICATION_SYSTEM.format(
        name=profile.get("name", "Unknown"),
        age=profile.get("age", "Unknown"),
        gender=profile.get("gender", "Unknown"),
        chief_complaint=profile.get("chief_complaint", "Unknown"),
    )
    user_prompt = RELEVANCE_VERIFICATION_USER.format(
        question=question,
        answer=answer,
        answer_topic=answer_topic,
    )
    
    return system_prompt, user_prompt