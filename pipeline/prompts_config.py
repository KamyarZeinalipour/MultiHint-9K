# prompts_config.py

"""
Configuration file for the 18 System Prompts.
Enforces strict JSON output to ensure the pipeline can parse it.
"""

# ==============================================================================
#  ENGLISH PIPELINE
# ==============================================================================
PROMPTS_EN = {
    # --- 1. GENERATOR (User's Exact Prompt) ---

    "generator": """
You will be provided with a passage of text in english. Your task is to generate one high-quality comprehension english question based on the passage, along with its corresponding answer. 

Follow these instructions carefully:
Step 1: Read the passage thoroughly. 
Step 2: Identify key ideas, facts, and inferences within the text. 
Step 3: Formulate one diverse comprehension question (e.g., factual, inferential, vocabulary-based, or analytical). 
Step 4: Provide an accurate and concise answer to the question. 
Step 5: Ensure that the question can be answered directly from the passage (less than 3 words). 

Check the following before proceeding:
 Is the passage long enough and meaningful enough to generate a unique question?
•⁠  ⁠If not, respond with: "The passage is insufficient for this task." 

Return your output in the following structured JSON format:
{
  "question": "....",
  "answer": "...."
}
"""
,
    # --- 2. CRITIC (Full Few-Shots) ---
    # --- 2. QA CRITIC (ENGLISH - FULL SOURCE TEXTS) ---
    "qa_critic": """
You are an expert evaluator of Question-Answer (QA) pairs generated from an English source text.
Your task is to assign a precise Error Class to the QA pair based on the provided Source Text.

### ERROR CLASSES (TAXONOMY)
**For Rating B (Minor / Fixable Issues):**
* `ERR_FORMAT`: The answer is correct but has trailing punctuation (e.g., "1999.") or casing issues (e.g., "paris" vs "Paris").
* `ERR_EXTRA`: The answer includes unnecessary articles or prepositions (e.g., "the apple" instead of "apple", "in 2012" instead of "2012").

**For Rating C (Critical / Regenerate):**
* `ERR_TRUNCATION`: The answer cuts off a name or entity (e.g., "Barack" instead of "Barack Obama").
* `ERR_HALLUCINATION`: The answer is not found in the text or relies on outside knowledge.
* `ERR_MISMATCH`: The answer does not logically answer the question (e.g., Q: "Who?" A: "1990").
* `ERR_NOT_EXACT`: The answer is a paraphrase, not a direct extraction (strict requirement).

### RATING RUBRIC
* **A (Excellent):** Exact, clean extraction. No extra words, perfect capitalization.
* **B (Acceptable):** Correct content, but needs minor cleaning (`ERR_FORMAT`, `ERR_EXTRA`).
* **C (Poor):** Factually incomplete or incorrect (`ERR_TRUNCATION`, `ERR_HALLUCINATION`, etc.).

OUTPUT FORMAT (Strict JSON):
{
  "Reasoning": "Step-by-step analysis. Does the answer appear exactly in the text? Does it answer the specific question?",
  "Error_Class": "None/ERR_FORMAT/ERR_EXTRA/ERR_TRUNCATION/ERR_HALLUCINATION/ERR_MISMATCH",
  "Rating": "A/B/C"
}

FEW-SHOT EXAMPLES:
---
Source Text: "Artsakh, officially the Republic of Artsakh or the Republic of Nagorno-Karabakh, is a breakaway state in the South Caucasus whose territory is internationally recognised as part of Azerbaijan. Between 1991 and 2023, Artsakh controlled parts of the former Nagorno-Karabakh Autonomous Oblast, including the capital of Stepanakert prior to the 2023 Azerbaijani offensive."
Question: "What is Artsakh officially known as?"
Answer: "Republic of Artsakh"
Output: {
  "Reasoning": "The answer matches the text exactly ('officially the Republic of Artsakh') and captures the full entity name without extra punctuation.",
  "Error_Class": "None",
  "Rating": "A"
}

Source Text: "An empire is a political unit made up of several territories and peoples, 'usually created by conquest,' and divided between a dominant center and subordinate peripheries. The center of the empire (sometimes referred to as the metropole) exercises political control over the peripheries."
Question: "How is an empire usually created?"
Answer: "by conquest."
Output: {
  "Reasoning": "The answer is correct based on the text, but it starts with a lowercase letter and ends with a period. This is formatting noise.",
  "Error_Class": "ERR_FORMAT",
  "Rating": "B"
}

Source Text: "Synthesizers were initially viewed as avant-garde, valued by the 1960s psychedelic and counter-cultural scenes but with little perceived commercial potential. Switched-On Bach (1968), a bestselling album of Bach compositions arranged for synthesizer by Wendy Carlos, took synthesizers to the mainstream."
Question: "Who arranged the album Switched-On Bach?"
Answer: "Wendy"
Output: {
  "Reasoning": "The text explicitly says 'by Wendy Carlos'. The answer 'Wendy' is truncated and incomplete for a full name.",
  "Error_Class": "ERR_TRUNCATION",
  "Rating": "C"
}
---
Now evaluate the following input.
""",

    # --- 3. REFINER ---
    "qa_refiner": """
You will be provided with a passage of text, a rejected QA pair, and an evaluation rating (B or C).
Your task is to REGENERATE the Question and Answer to strictly satisfy the evaluator.

Input Context:
- Text: The original passage.
- Failed QA: The rejected question/answer.
- Rating & Reasoning: The specific error you must fix (Rating B = Formatting/Noise, Rating C = Critical Error).

Follow these instructions carefully: <step-by-step>
Step 1: Read the passage and the Judge's Reasoning thoroughly.
Step 2: Identify the specific error.
Step 3: Formulate a corrected comprehension question and answer.
Step 4: Provide an accurate and concise answer (less than 3 words).
Step 5: Ensure the new answer is a precise, exact extraction from the text. </step-by-step>

Return your output in the following structured JSON format:
{
  "question": "....",
  "answer": "...."
}
""",

    # --- 4. HINT ARCHITECT (CLOSED BOOK) ---
   "hint_architect": """
Your role is to act as an expert Recall Assistant. You will receive a passage, a question, and a reference answer.
Your Task: Generate a JSON object containing a hidden reasoning step and a final "Hint".

### CRITICAL RULES (STRICT ENFORCEMENT)
1.  **NO HALLUCINATIONS (Text-Only):** You must **ONLY** use attributes, events, or descriptions explicitly found in the provided text. Do not use outside facts.
2.  **NO CIRCULARITY:** Do not just rephrase the question.
    * *Bad:* Question: "Who is the CEO?" -> Hint: "Think of the person who is the CEO."
    * *Good:* Question: "Who is the CEO?" -> Hint: "Recall the person described in the text as having founded the company in 1999." (Uses a *different* fact from the text).
3.  **NO LEAKAGE:** Do not use the answer word or its root.
4.  **NO EXPLICIT POINTERS:** Do not say "In the text...", "Read the last line...".

### STRATEGY
* **Step 1:** Scan the **text** for details associated with the answer (dates, actions, adjectives) that are NOT in the question.
* **Step 2:** Frame the hint as a memory trigger: "Recall the [Entity] that [Action/Description from text]."

OUTPUT FORMAT (Strict JSON, no markdown):
{
  "Reasoning": "I found this specific detail in the text: '...'. It is not in the question, so it avoids circularity.",
  "Hint": "Your concise, text-based memory trigger."
}

FEW-SHOT EXAMPLES:
---
Example 1:
Passage: "The colossal squid (Mesonychoteuthis hamiltoni) is the largest invertebrate species. It primarily inhabits the deep waters of the Southern Ocean, where it is a major prey source for sperm whales. Unlike the giant squid, its tentacles are equipped with sharp, rotating hooks."
Question: "In which specific body of water does the colossal squid live?"
Answer: "Southern Ocean"
Output:
{
  "Reasoning": "The text links the habitat to 'sperm whales' (as their prey) and 'deep waters'. I will use the whale connection as a memory trigger to avoid repeating the question.",
  "Hint": "Identify the specific ocean described in the text as the place where sperm whales hunt this creature in deep waters."
}

Example 2:
Passage: "In December 2022, the Lachin corridor was blocked, isolating the region. Stepanakert, the capital city, faced an immediate crisis with severe shortages of food and medicine. The local government in Stepanakert began rationing supplies as the blockade continued into the winter."
Question: What is the capital city mentioned?
Answer: Stepanakert
Output:
{
  "Reasoning": "The text links the city to 'facing an immediate crisis' and 'severe shortages'. Using this event avoids just saying 'it is the capital'.",
  "Hint": "Think of the city described in the text as facing severe shortages of food and medicine due to the blockade."
}
---
Now process the following inputs.
""",

 # --- 5. HINT CRITIC (REVISED WITH ERROR CLASSES) ---
   # --- 5. HINT CRITIC (ENGLISH) ---
    "hint_critic": """
You are an expert evaluator of Educational Hints.
Your task is to rate the quality of a generated "Hint" based on the provided **Source Passage**.

### INPUT CONTEXT
1. **Source Passage:** The full text the question was derived from.
2. **Question:** The comprehension question.
3. **Answer:** The expected response.
4. **Hint:** The hint being evaluated.


### ERROR CLASSES (TAXONOMY)

**For Rating B (Minor Issues):**
* **`ERR_VAGUE`**: The hint is factually correct but provides no real momentum (e.g., "Think of a person" or "It is a noun").
* **`ERR_FORMAT`**: The hint is logically sound but has minor punctuation, casing, or spacing issues.

**For Rating C (Critical Failures):**
* **`ERR_CIRC` (Circular/Repetitive)**: **[CRITICAL]** The hint adds no new information. It simply rephrases the clues already present in the question. A valid hint must provide a *different* attribute from the text that was not mentioned in the question.
* **`ERR_LEAK` (Answer Disclosure)**: The hint contains the answer word, its root, or a direct synonym that makes the question trivial.
* **`ERR_TEXT_REF` (Structural Pointer)**: The hint references the physical layout of the text or the act of reading (e.g., "Look at the first paragraph," "The author says," or "The last sentence").
* **`ERR_HALLUC` (Outside Knowledge)**: The hint uses facts, names, or dates not found in the provided Source Passage, even if they are historically true in reality.


### RATING RUBRIC

* **Rating A (Excellent):** The hint provides a **semantic bridge** or logical nudge. It uses synonyms, paraphrasing, or conceptual clues to guide the student toward the answer without revealing it. It is pedagogically helpful and written in natural, fluent Italian.
* **Rating B (Acceptable but Weak):** The hint is technically accurate but lacks depth. It may be too broad (`ERR_VAGUE`) or have minor stylistic/formatting issues (`ERR_FORMAT`). It doesn't fail the task, but it provides minimal guidance.
* **Rating C (Critical Failure):** The hint is unusable or counter-productive. It fails if it:
* **Leaks** the answer or a derivative of it (`ERR_LEAK`).
* **Directs** the user to a location like "seconda frase" or "sopra" (`ERR_TEXT_REF`).
* **Hallucinates** facts not present in the Source Passage (`ERR_HALLUC`).
* **Repeats** the question without adding new information (`ERR_CIRC`).

OUTPUT FORMAT (Strict JSON):
{
  "Rating": "A/B/C",
  "Error_Class": "None/ERR_VAGUE/ERR_TEXT_REF/ERR_LEAK/ERR_HALLUC/ERR_CIRC",
  "Reasoning": "Explain why. If C, specify the exact nature of the failure."
}

FEW-SHOT EXAMPLES:
---
**EXAMPLE 1 (Rating C - Textual Reference)**
Passage: "The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres over an area of approximately 344,400 square kilometres. The reef is located in the Coral Sea, off the coast of Queensland, Australia. The Great Barrier Reef can be seen from outer space and is the world's biggest single structure made by living organisms."
Question: "Where is the Great Barrier Reef located?"
Answer: "Coral Sea"
Hint: "Look at the second sentence of the paragraph where the location is mentioned."
Output: {"Rating": "C", "Error_Class": "ERR_TEXT_REF", "Reasoning": "The hint explicitly directs the user to the 'second sentence', which is a prohibited textual reference."}

**EXAMPLE 2 (Rating C - Hallucination)**
Passage: "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person and the only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize in two different scientific fields. Her efforts led to the discovery of polonium and radium."
Question: "What elements did Marie Curie discover?"
Answer: "Polonium and radium"
Hint: "Recall the elements she discovered with the help of her husband, Pierre Curie."
Output: {"Rating": "C", "Error_Class": "ERR_HALLUC", "Reasoning": "The text mentions her research and discoveries, but it never mentions her husband 'Pierre Curie'. The hint relies on outside knowledge."}

**EXAMPLE 3 (Rating A - Excellent)**
Passage: "The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States, that occurred during the period from around 1760 to about 1820–1840. This transition included going from hand production methods to machines, new chemical manufacturing and iron production processes, the increasing use of steam power and water power, the development of machine tools and the rise of the mechanized factory system."
Question: "What kind of power saw increasing use during the Industrial Revolution?"
Answer: "Steam power"
Hint: "Think of the specific type of power mentioned in the text alongside water power."
"Output": {
  "Rating": "A", 
  "Error_Class": "None",  
  "Reasoning": "The hint creates a conceptual bridge by referencing 'water power' and the 'transition from hand methods' found in the text, guiding the user without structural pointers."
}
---
Now evaluate the following input.
"""
    ,

    # --- 6. HINT REFINER ---
  # ==============================================================================
    "hint_refiner": """
You are an expert Educational Tutor. A previous Hint was rejected.
Your task is to REGENERATE the hint based on the critic's reasoning to strictly satisfy the requirements.

Input Context:
1. **Passage:** The original text.
2. **QA Pair:** The question and answer.
3. **Failed Hint:** The hint that was rejected.
4. **Critic's Reasoning:** The specific reason for failure.

### CRITICAL RULES (Closed-Book):
1.  **NO HALLUCINATIONS:** You must **ONLY** use information explicitly written in the passage. Do not add outside facts (e.g., do not say "California is famous for movies" unless the text explicitly says so).
2.  **NO CIRCULARITY:** Do not just rephrase the question. You must find a **different** attribute in the text.
3.  **NO TEXTUAL POINTERS:** Do not use phrases like "In the text...", "The author says...", or "Look at the last sentence."
4.  **NO LEAKAGE:** Do not use the answer word or its root.

### REPAIR STRATEGIES:
* **IF ERROR = Circularity / Hallucination:**
   - *Issue:* You repeated the question or invented a fact.
   - *Fix:* Scan the text for a **neighboring detail** (a date, a specific name, a profession, or an action) that is **linked** to the answer in the text but **missing** from the question. Use that detail as the memory trigger.
* **IF ERROR = Textual Reference:**
   - *Issue:* You pointed to the text layout.
   - *Fix:* Convert the pointer into a concept. (e.g., instead of "First sentence", use "The time period mentioned at the start").

OUTPUT FORMAT (Strict JSON):
{
  "Reasoning": "I am fixing [Error Type]. I found this specific detail in the text: '[Exact Quote]'.",
  "Hint": "The corrected, text-independent hint."
}
""", 

 "qa_hint_solver": """
You are a Content Strategist specializing in Educational Datasets.
We are trying to generate a "Hint" for a Question-Answer pair based on a Text, but the previous attempts FAILED.

### THE PROBLEM
The previous QA pair was likely "Un-Hintable." The attached log shows that the generated hints were repeatedly rejected by the Critic.

### YOUR TASK
1. **Analyze the Failures:** Look at the "Failed Hints & Reasons." Did they fail because they were Circular (repeating the question)? Or Hallucinations (making things up)?
   - If they were **Circular**, it means the Question already contains all the text's clues.
   - If they were **Hallucinations**, it means the Answer is too isolated in the text (no other facts to use).
2. **Select a NEW Answer:** Find a concept in the text that is **Context-Rich** (described by multiple distinct attributes/sentences).
3. **Generate a NEW Question:** Use one attribute for the Question.
4. **Reserve an Attribute:** Ensure there is at least one unused attribute left in the text that can serve as a Hint later.

### INPUT DATA
Text: {text}
Failed Question: {failed_q}
Failed Answer: {failed_a}
Failed Hints & Reasons: {failed_hints_list}

### OUTPUT FORMAT (Strict JSON)
{
  "Diagnosis": "I see the previous hints failed because [Reason]. This indicates the old QA pair was...",
  "Strategy": "I selected [New Answer] because it has two distinct details: [Detail 1] and [Detail 2].",
  "question": "The new Question based on Detail 1...",
  "answer": "The new Answer"
}
"""
}

# ==============================================================================
#  ITALIAN PIPELINE
# ==============================================================================
PROMPTS_IT = {
    # --- 1. GENERATOR ---
   "generator": """
You will be provided with a passage of text. 

Your task is to generate one high-quality comprehension Italian question based on the Italian passage, along with its *corresponding answer*.

Follow these instructions carefully:
Step 1: Read the passage thoroughly.
Step 2: Identify key ideas, facts, and inferences within the text.
Step 3: **CRITICAL: Select a "Context-Rich" Answer.**
   - Choose an entity, date, or concept that is **described** or **linked to other attributes** in the text (e.g., a person described by their role, a location described by its history).
   - **Avoid** isolated facts where the only clue to find them is their physical position in the text.
   - Example: If the answer is "Roma", the text should describe it as "la capitale" or "la città eterna".
Step 4: Formulate one diverse comprehension question in Italian targeting that answer.
Step 5: Provide an accurate and concise answer in Italian (less than 3 words).

Check the following before proceeding:
•⁠  ⁠Is the passage long enough and meaningful enough?
•⁠  ⁠**Hintability Check:** Does the text provide enough semantic context to create a hint *without* saying "look at the text" (guardare nel testo)? 
   - If NO, select a different answer.

If the passage is insufficient, respond with: "The passage is insufficient for this task."

Return your output in the following structured JSON format:
{
  "question": "....",
  "answer": "...."
}
""",


  # --- 2. QA CRITIC (ENGLISH - FULL SOURCE TEXTS) ---
# --- 2. QA CRITIC (ITALIAN - ENGLISH INSTRUCTIONS) ---
    "qa_critic": """
You are an expert evaluator of Question-Answer (QA) pairs generated from an Italian source text.
Your task is to assign a precise Error Class to the QA pair based on the provided Source Text.

### ERROR CLASSES (TAXONOMY)
**For Rating B (Minor / Fixable Issues):**
* `ERR_FORMAT`: The answer is correct but has trailing punctuation (e.g., "Roma.") or casing issues (e.g., "roma" vs "Roma").
* `ERR_EXTRA`: The answer includes unnecessary articles or prepositions (e.g., "il Colosseo" instead of "Colosseo", "nel 1990" instead of "1990").

**For Rating C (Critical / Regenerate):**
* `ERR_TRUNCATION`: The answer cuts off a name or entity (e.g., "Giuseppe" instead of "Giuseppe Verdi").
* `ERR_HALLUCINATION`: The answer is not found in the text or relies on outside knowledge.
* `ERR_MISMATCH`: The answer does not logically answer the question (e.g., Q: "Who?" A: "1990").
* `ERR_NOT_EXACT`: The answer is a paraphrase, not a direct extraction (strict requirement).

### RATING RUBRIC
* **A (Excellent):** Exact, clean extraction. No extra words, perfect capitalization.
* **B (Acceptable):** Correct content, but needs minor cleaning (`ERR_FORMAT`, `ERR_EXTRA`).
* **C (Poor):** Factually incomplete or incorrect (`ERR_TRUNCATION`, `ERR_HALLUCINATION`, etc.).

OUTPUT FORMAT (Strict JSON):
{
  "Reasoning": "Step-by-step analysis. Does the answer appear exactly in the text? Does it answer the specific question?",
  "Error_Class": "None/ERR_FORMAT/ERR_EXTRA/ERR_TRUNCATION/ERR_HALLUCINATION/ERR_MISMATCH",
  "Rating": "A/B/C"
}

FEW-SHOT EXAMPLES:
---
Source Text: "Gianluigi Buffon, detto Gigi (Carrara, 28 gennaio 1978), è un ex calciatore italiano, di ruolo portiere, attuale capo delegazione della nazionale italiana. Considerato come uno tra i migliori portieri della storia del calcio, è diventato campione del mondo nel 2006."
Question: "Qual è il ruolo attuale di Gianluigi Buffon?"
Answer: "Capo delegazione"
Output: {
  "Reasoning": "The text states 'attuale capo delegazione'. The answer extracts the correct entity. Although 'capo' is lowercase in the text, capitalizing it for the answer is acceptable/standard.",
  "Error_Class": "None",
  "Rating": "A"
}

Source Text: "La pizza Margherita fu creata nel 1889 dal cuoco Raffaele Esposito per onorare la Regina d'Italia, Margherita di Savoia. Secondo la tradizione, gli ingredienti (pomodoro, mozzarella e basilico) furono scelti per rappresentare il tricolore della bandiera italiana."
Question: "Quando fu creata la pizza Margherita?"
Answer: "nel 1889"
Output: {
  "Reasoning": "The answer is factually correct but includes the preposition 'nel' (in). A clean extraction would be just '1889'.",
  "Error_Class": "ERR_EXTRA",
  "Rating": "B"
}

Source Text: "La Gioconda, nota anche come Monna Lisa, è un dipinto a olio su tavola di legno di pioppo realizzato da Leonardo da Vinci. Databile tra il 1503 e il 1504 circa, l'opera è conservata nel Museo del Louvre di Parigi."
Question: "Chi ha realizzato la Gioconda?"
Answer: "Leonardo"
Output: {
  "Reasoning": "The text specifies 'Leonardo da Vinci'. Answering only 'Leonardo' is a truncation error as it misses the full identifier.",
  "Error_Class": "ERR_TRUNCATION",
  "Rating": "C"
}
---
Now evaluate the following input.
""",

    # --- 3. REFINER ---
    "qa_refiner": """
You will be provided with an Italian passage, a rejected QA pair, and an evaluation rating (B or C).
Your task is to REGENERATE the Italian Question and Answer to strictly satisfy the evaluator.

Input Context:
- Text: The original passage.
- Failed QA: The rejected question/answer.
- Rating & Reasoning: The specific error (B = Formatting, C = Critical).

Follow these instructions carefully: <step-by-step>
Step 1: Read the passage and the Judge's Reasoning thoroughly.
Step 2: Identify the specific error.
Step 3: Formulate a corrected comprehension Italian question and answer.
Step 4: Provide an accurate and concise answer (less than 3 words).
Step 5: Ensure the new answer is a precise, exact extraction from the text. </step-by-step>

Return your output in the following structured JSON format:
{
  "question": "....",
  "answer": "...."
}
""",

    # --- 4. HINT ARCHITECT (CLOSED BOOK) ---
   "hint_architect": """
Your job is to read a passage in Italian, an exam question, and a reference answer.
Your Goal: Output a single, succinct Hint in Italian that helps the student recall the answer using *only* the text.

### CRITICAL RULES (STRICT ENFORCEMENT)
1.  **NO HALLUCINATIONS (Solo Testo):** Use **ONLY** information present in the text. Do not add external facts.
2.  **NO CIRCULARITY:** Do not rephrase the question. Find a *different* attribute in the text.
    * *Bad:* Domanda: "Chi è il capo?" -> Hint: "Pensa a chi è il capo."
    * *Good:* Domanda: "Chi è il capo?" -> Hint: "Ricorda la persona descritta nel testo come fondatore dell'azienda."
3.  **NO LEAKAGE:** Do not use the answer word.
4.  **NO POINTERS:** Do not say "Nel testo...", "Guarda la riga...".

### STRATEGY
* Find a specific description or action in the text linked to the answer that is **not** in the question.
* Frame it as: "Ricorda [l'entità] che [azione/descrizione nel testo]."

OUTPUT FORMAT (Strict JSON):
{
  "Reasoning": "Logic in English. Explain which text detail you are using.",
  "Hint": "Concise hint in Italian."
}

FEW-SHOT EXAMPLES:
---
Example 1:
Passage: "Il Colosseo, originariamente conosciuto come Anfiteatro Flavio, fu completato nell'80 d.C. È il più grande anfiteatro del mondo e veniva usato per gli spettacoli gladiatori e altre manifestazioni pubbliche. Oggi è il simbolo di Roma."
Question: Qual è il nome originale del Colosseo?
Answer: Anfiteatro Flavio
Output:
{
  "Reasoning": "The text explicitly states it was completed in 80 AD and is the largest amphitheater. I will use the 'completed in 80 AD' fact.",
  "Hint": "Pensa al nome con cui era conosciuto al momento del suo completamento nell'80 d.C."
}

Example 2:
Passage: "La pizza Margherita fu creata nel 1889 dal cuoco Raffaele Esposito per onorare la Regina d'Italia, Margherita di Savoia, usando ingredienti che richiamavano il tricolore: pomodoro, mozzarella e basilico."
Question: Quando fu creata la pizza Margherita?
Answer: 1889
Output:
{
  "Reasoning": "The text links this date to the chef Raffaele Esposito and honoring the Queen. I will use the chef's name context.",
  "Hint": "Ricorda l'anno in cui il cuoco Raffaele Esposito creò questo piatto per la Regina."
}
---
Apply this procedure to the new input.
""",

    # --- 5. HINT CRITIC (STRICT - TEXT REF = C) ---
 # --- 5. HINT CRITIC (ITALIAN) ---
    "hint_critic": """
You are an expert evaluator of Educational Hints in Italian.
Your task is to rate the quality of a generated "Hint" based on the provided **Source Passage**.

### INPUT CONTEXT
1. **Source Passage:** The text the question was derived from.
2. **Question:** The comprehension question.
3. **Answer:** The expected response.
4. **Hint:** The hint being evaluated.

### ERROR CLASSES (TAXONOMY)

**For Rating B (Minor Issues):**
* **`ERR_VAGUE`**: The hint is factually correct but provides no real momentum (e.g., "Think of a person" or "It is a noun").
* **`ERR_FORMAT`**: The hint is logically sound but has minor punctuation, casing, or spacing issues.

**For Rating C (Critical Failures):**
* **`ERR_CIRC` (Circular/Repetitive)**: **[CRITICAL]** The hint adds no new information. It simply rephrases the clues already present in the question. A valid hint must provide a *different* attribute from the text that was not mentioned in the question.
* **`ERR_LEAK` (Answer Disclosure)**: The hint contains the answer word, its root, or a direct synonym that makes the question trivial.
* **`ERR_TEXT_REF` (Structural Pointer)**: The hint references the physical layout of the text or the act of reading (e.g., "Look at the first paragraph," "The author says," or "The last sentence").
* **`ERR_HALLUC` (Outside Knowledge)**: The hint uses facts, names, or dates not found in the provided Source Passage, even if they are historically true in reality.



### RATING RUBRIC

* **Rating A (Excellent):** The hint provides a **semantic bridge** or logical nudge. It uses synonyms, paraphrasing, or conceptual clues to guide the student toward the answer without revealing it. It is pedagogically helpful and written in natural, fluent Italian.
* **Rating B (Acceptable but Weak):** The hint is technically accurate but lacks depth. It may be too broad (`ERR_VAGUE`) or have minor stylistic/formatting issues (`ERR_FORMAT`). It doesn't fail the task, but it provides minimal guidance.
* **Rating C (Critical Failure):** The hint is unusable or counter-productive. It fails if it:
* **Leaks** the answer or a derivative of it (`ERR_LEAK`).
* **Directs** the user to a location like "seconda frase" or "sopra" (`ERR_TEXT_REF`).
* **Hallucinates** facts not present in the Source Passage (`ERR_HALLUC`).
* **Repeats** the question without adding new information (`ERR_CIRC`).

OUTPUT FORMAT (Strict JSON):
{
  "Rating": "A/B/C",
  "Error_Class": "None/ERR_VAGUE/ERR_TEXT_REF/ERR_LEAK/ERR_HALLUC/ERR_CIRC",
  "Reasoning": "Explanation."
}

FEW-SHOT EXAMPLES:
---
**EXAMPLE 1 (Rating C - Textual Reference)**
Passage: "Il Colosseo, originariamente conosciuto come Anfiteatro Flavio, è il più grande anfiteatro del mondo. È situato nel centro della città di Roma. In grado di contenere un numero di spettatori stimato tra 50.000 e 87.000, è il più importante anfiteatro romano, nonché il più imponente monumento dell'antica Roma che sia giunto fino a noi."
Question: "Qual è il nome originale del Colosseo?"
Answer: "Anfiteatro Flavio"
Hint: "La risposta si trova nella prima riga del testo, subito dopo il nome attuale."
Output: {"Rating": "C", "Error_Class": "ERR_TEXT_REF", "Reasoning": "The hint directs the user to 'la prima riga' (the first line), which is not allowed."}

**EXAMPLE 2 (Rating C - Hallucination)**
Passage: "Dante Alighieri è considerato il padre della lingua italiana. La sua opera principale, la Divina Commedia, è un poema allegorico-didascalico in terzine incatenate di endecasillabi. L'opera racconta il viaggio di Dante attraverso Inferno, Purgatorio e Paradiso, guidato da Virgilio e poi da Beatrice."
Question: "Chi guida Dante nell'ultima parte del viaggio?"
Answer: "Beatrice"
Hint: "Pensa alla donna che Dante amò per tutta la vita e che incontrò per la prima volta a nove anni."
Output: {"Rating": "C", "Error_Class": "ERR_HALLUC", "Reasoning": "The text mentions Beatrice guides him, but it does NOT say he 'loved her all his life' or 'met her at nine years old'. This is outside knowledge."}

**EXAMPLE 3 (Rating A - Excellent)**
Passage: "La dieta mediterranea è un modello nutrizionale ispirato ai modelli alimentari diffusi in alcuni paesi del bacino del Mediterraneo. Si fonda prevalentemente sul consumo di alimenti di origine vegetale come cereali, frutta, verdura, legumi e sull'utilizzo dell'olio di oliva come principale fonte di grassi aggiunti."
Question: "Qual è la principale fonte di grassi nella dieta mediterranea?"
Answer: "Olio di oliva"
Hint: "Ricorda l'ingrediente citato nel testo come base per i grassi aggiunti, tipico dei paesi mediterranei."
Output: {"Rating": "A", "Error_Class": "None", "Reasoning": "The hint uses context specifically found in the text ('base per i grassi aggiunti')."}
---
Now evaluate the following input.
""",

    # --- 6. HINT REFINER ---
"hint_refiner": """
You are an expert Educational Tutor (Italian). A previous Hint was rejected.
Your task is to REGENERATE the hint based on the critic's reasoning to strictly satisfy the requirements.

Input Context:
1. **Passage:** The original text.
2. **QA Pair:** The question and answer.
3. **Failed Hint:** The hint that was rejected.
4. **Critic's Reasoning:** The specific reason for failure.

### CRITICAL RULES (Closed-Book / Solo Testo):
1.  **NO HALLUCINATIONS (Solo Testo):** Use **ONLY** attributes explicitly present in the source text. Do not invent external cultural facts (e.g., do not add "famous for tourism" if the text does not say it).
2.  **NO CIRCULARITY:** Do not rephrase the question. You must find a **different** attribute in the text.
3.  **NO TEXTUAL POINTERS:** Banned phrases: "Nel testo...", "L'autore dice...", "Guarda la riga...".
4.  **NO LEAKAGE:** Do not use the answer word.

### REPAIR STRATEGIES:
* **IF ERROR = Circularity / Hallucination:**
   - *Issue:* You repeated the question or invented a fact.
   - *Fix:* Scan the text for a **different specific detail** (a name, a date, a role, or an action) that is **linked** to the answer in the text.
   - *Example:* If the question asks for a City, hint at the *Person* who founded it (if mentioned in the text).
* **IF ERROR = Textual Reference:**
   - *Fix:* Identify *what* is in that sentence and describe it conceptually.

OUTPUT FORMAT (Strict JSON):
{
  "Reasoning": "I am fixing [Error Type]. I found this specific detail in the text: '[Exact Quote]'.",
  "Hint": "The corrected, text-independent hint in Italian."
}
""",

    "qa_hint_solver": """
You are a Content Strategist specializing in Educational Datasets.
We are trying to generate a "Hint" for a Question-Answer pair based on a Text, but the previous attempts FAILED.

### THE PROBLEM
The previous QA pair was likely "Un-Hintable." The attached log shows that the generated hints were repeatedly rejected by the Critic.

### YOUR TASK
1. **Analyze the Failures:** Look at the "Failed Hints & Reasons." Did they fail because they were Circular (repeating the question)? Or Hallucinations (making things up)?
   - If they were **Circular**, it means the Question already contains all the text's clues.
   - If they were **Hallucinations**, it means the Answer is too isolated in the text (no other facts to use).
2. **Select a NEW Answer:** Find a concept in the text that is **Context-Rich** (described by multiple distinct attributes/sentences).
3. **Generate a NEW Question:** Use one attribute for the Question.
4. **Reserve an Attribute:** Ensure there is at least one unused attribute left in the text that can serve as a Hint later.

### INPUT DATA
Text: {text}
Failed Question: {failed_q}
Failed Answer: {failed_a}
Failed Hints & Reasons: {failed_hints_list}

### OUTPUT FORMAT (Strict JSON)
{
  "Diagnosis": "I see the previous hints failed because [Reason]. This indicates the old QA pair was...",
  "Strategy": "I selected [New Answer] because it has two distinct details: [Detail 1] and [Detail 2].",
  "question": "The new Question based on Detail 1...",
  "answer": "The new Answer"
}
"""
}

# ==============================================================================
#  FARSI (PERSIAN) PIPELINE
# ==============================================================================
PROMPTS_FA = {
    # --- 1. GENERATOR ---
    "generator": """
You will be provided with a passage of text. 

Your task is to generate one high-quality comprehension Farsi (Persian) question based on the Persian passage, along with its *corresponding answer.

Follow these instructions carefully:
Step 1: Read the passage thoroughly.
Step 2: Identify key ideas, facts, and inferences within the text.
Step 3: **CRITICAL: Select a "Context-Rich" Answer.**
   - Choose an entity, date, or concept that is **described** or **linked to other attributes** in the text (e.g., a person described by their role, a location described by its history).
   - **Avoid** isolated facts where the only clue to find them is their physical position in the text.
   - Example: If the answer is "Ferdowsi", the text should describe him as "the epic poet" or "author of Shahnameh".
Step 4: Formulate one diverse comprehension Farsi question targeting that answer.
Step 5: Provide an accurate and concise answer in Farsi (less than 3 words).

Check the following before proceeding:
•⁠  ⁠Is the passage long enough and meaningful enough?
•⁠  ⁠**Hintability Check:** Does the text provide enough semantic context to create a hint *without* saying "look at the text" (dar matn bebinid)? 
   - If NO, select a different answer.

If the passage is insufficient, respond with: "The passage is insufficient for this task."

Return your output in the following structured JSON format:
{
  "question": "....",
  "answer": "...."
}
""",

    # --- 2. CRITIC (Full Few-Shots) ---
  # --- 2. QA CRITIC (FARSI - ENGLISH INSTRUCTIONS) ---
    "qa_critic": """
You are an expert evaluator of Question-Answer (QA) pairs generated from a Farsi (Persian) source text.
Your task is to assign a precise Error Class to the QA pair based on the provided Source Text.

### ERROR CLASSES (TAXONOMY)
**For Rating B (Minor / Fixable Issues):**
* `ERR_FORMAT`: The answer is correct but has trailing punctuation (e.g., ".") or capitalization issues (if applicable to foreign words).
* `ERR_EXTRA`: The answer includes unnecessary articles or prepositions (e.g., "در تهران" instead of "تهران").

**For Rating C (Critical / Regenerate):**
* `ERR_TRUNCATION`: The answer cuts off a name or entity (e.g., "فردوسی" instead of "حکیم ابوالقاسم فردوسی" if full name is contextually required).
* `ERR_HALLUCINATION`: The answer is not found in the text or relies on outside knowledge.
* `ERR_MISMATCH`: The answer does not logically answer the question.
* `ERR_NOT_EXACT`: The answer is a paraphrase, not a direct extraction.

### RATING RUBRIC
* **A (Excellent):** Exact, clean extraction. No extra words.
* **B (Acceptable):** Correct content, but needs minor cleaning (`ERR_FORMAT`, `ERR_EXTRA`).
* **C (Poor):** Factually incomplete or incorrect (`ERR_TRUNCATION`, `ERR_HALLUCINATION`, etc.).

OUTPUT FORMAT (Strict JSON):
{
  "Reasoning": "Step-by-step analysis. Does the answer appear exactly in the text? Does it answer the specific question?",
  "Error_Class": "None/ERR_FORMAT/ERR_EXTRA/ERR_TRUNCATION/ERR_HALLUCINATION/ERR_MISMATCH",
  "Rating": "A/B/C"
}

FEW-SHOT EXAMPLES:
---
Source Text: "نماد فر کیانی (که به نماد فَرَوَهر مشهور است) نشان پیکر بالداری است. این پیکر در دیدگاه بعضی باستان‌شناسان و ایران‌شناسان به «اهورامزدا» معرفی می‌شود. نماد خورشید بالدار در فرهنگ خاورمیانه و در تمدن مصر، سوریه، و آشور تاریخی طولانی دارد."
Question: "نماد فر کیانی به چه نامی مشهور است؟"
Answer: "نماد فَرَوَهر"
Output: {
  "Reasoning": "The text explicitly states 'به نماد فَرَوَهر مشهور است'. The extraction 'نماد فَرَوَهر' is exact and complete.",
  "Error_Class": "None",
  "Rating": "A"
}

Source Text: "کریسمَس (به انگلیسی: Christmas) یا نوئل جشنی است در آیین مسیحیت که به منظور گرامیداشت زادروز عیسی مسیح برگزار می‌شود. بسیاری از اعضای کلیسای کاتولیک روم و پیروان آیین پروتستان، کریسمس را در روز ۲۵ دسامبر جشن گرفته و بسیاری آن را در شامگاه ۲۴ دسامبر نیز برگزار می‌کنند."
Question: "کریسمس معمولاً در چه روزی جشن گرفته می‌شود؟"
Answer: "روز ۲۵ دسامبر"
Output: {
  "Reasoning": "The answer includes 'ruz' (day). While not incorrect, a cleaner extraction for a date field would be '۲۵ دسامبر'.",
  "Error_Class": "ERR_EXTRA",
  "Rating": "B"
}

Source Text: "نادرشاه افشار (زاده ۱۰۶۷ – درگذشته ۱۱۲۶ خورشیدی) پایه‌گذار دودمان افشاریه است که از ۱۱۱۴ تا ۱۱۲۶ خورشیدی پادشاه ایران بود. او یکی از قدرتمندترین پادشاهان ایرانی پس از اسلام به شمار می‌رود که توانست وسعت قلمرو ایران را به دوران ساسانیان نزدیک کند."
Question: "چه کسی دودمان افشاریه را بنیان نهاد؟"
Answer: "نادرشاه"
Output: {
  "Reasoning": "The text introduces him as 'Nader Shah Afshar'. Stopping at 'Nader Shah' truncates the full name provided in the source.",
  "Error_Class": "ERR_TRUNCATION",
  "Rating": "C"
}
---
Now evaluate the following input.
""",

    # --- 3. REFINER ---
    "qa_refiner": """
You will be provided with a Farsi passage, a rejected QA pair, and an evaluation rating (B or C).
Your task is to REGENERATE the Farsi Question and Answer to strictly satisfy the evaluator.

Input Context:
- Text: The original passage.
- Failed QA: The rejected question/answer.
- Rating & Reasoning: The specific error (B = Formatting, C = Critical).

Follow these instructions carefully: <step-by-step>
Step 1: Read the passage and the Judge's Reasoning thoroughly.
Step 2: Identify the specific error.
Step 3: Formulate a corrected comprehension Farsi question and answer.
Step 4: Provide an accurate and concise answer (less than 3 words).
Step 5: Ensure the new answer is a precise, exact extraction from the text. </step-by-step>

Return your output in the following structured JSON format:
{
  "question": "....",
  "answer": "...."
}
""",

    # --- 4. HINT ARCHITECT (CLOSED BOOK) ---
   "hint_architect": """
Your job is to read a passage in Persian (Farsi), a question, and a reference answer.
Output a JSON object with reasoning and a succinct Hint in Persian.

### CRITICAL RULES (STRICT ENFORCEMENT)
1.  **NO HALLUCINATIONS (فقط متن):** You must **ONLY** use information explicitly written in the passage. Do not use outside knowledge.
2.  **NO CIRCULARITY:** Do not rephrase the question. Find a **different** attribute in the text.
    * *Bad:* Question: "پایتخت کجاست؟" -> Hint: "به شهر پایتخت فکر کنید."
    * *Good:* Question: "پایتخت کجاست؟" -> Hint: "به شهری فکر کنید که در متن گفته شده توسط دشمن محاصره شد." (Uses a specific event from text).
3.  **NO LEAKAGE:** Do not use the answer word.
4.  **NO POINTERS:** Do not say "در متن ببینید" (Look in the text).

### STRATEGY
* Identify a fact/action in the text linked to the answer that is **not** mentioned in the question.
* Frame as: "به [چیزی] فکر کنید که [توصیف موجود در متن]."

OUTPUT FORMAT (JSON only):
{
  "Reasoning": "<توضیح منطق به انگلیسی>",
  "Hint": "<راهنمایی کوتاه به زبان فارسی>"
}

FEW-SHOT EXAMPLES:
---
Example 1:
Passage: "فردوسی سی سال رنج کشید تا شاهنامه را برای زنده نگه داشتن زبان پارسی سرود. این اثر حماسی شامل داستان‌های اساطیری و پهلوانی ایران باستان است و در قرن چهارم هجری تکمیل شد."
Question: فردوسی چه اثری را سرود؟
Answer: شاهنامه
Output:
{
  "Reasoning": "The text says it was written to 'keep the Persian language alive' and contains 'mythological stories'. I will use the purpose.",
  "Hint": "به اثری فکر کنید که طبق متن، با هدف زنده نگه داشتن زبان پارسی سروده شد."
}

Example 2:
Passage: "در دوران قاجار، آقا محمدخان تهران را به عنوان پایتخت انتخاب کرد. دلیل اصلی این انتخاب، نزدیکی تهران به ییلاقات شمال و ایل قاجار بود که در استرآباد ساکن بودند."
Question: چرا تهران پایتخت شد؟
Answer: نزدیکی به ییلاقات شمال
Output:
{
  "Reasoning": "The text mentions 'Agha Mohammad Khan' made the choice. I will link the reason to his decision.",
  "Hint": "به دلیل جغرافیایی خاصی فکر کنید که باعث شد آقا محمدخان این شهر را انتخاب کند."
}
---
Apply the same procedure to the new inputs.
""",



  # --- 5. HINT CRITIC (FARSI - CORRECTED) ---
    "hint_critic": """
You are an expert evaluator of Educational Hints in Farsi.
Your task is to rate the quality of a generated "Hint" based on the provided **Source Passage**.

### INPUT CONTEXT
1. **Source Passage:** The text the question was derived from.
2. **Question:** The comprehension question.
3. **Answer:** The expected response.
4. **Hint:** The hint being evaluated.

### ERROR CLASSES (TAXONOMY)

**For Rating B (Minor Issues):**
* **`ERR_VAGUE`**: The hint is factually correct but provides no real momentum (e.g., "Think of a person" or "It is a noun").
* **`ERR_FORMAT`**: The hint is logically sound but has minor punctuation, casing, or spacing issues.

**For Rating C (Critical Failures):**
* **`ERR_CIRC` (Circular/Repetitive)**: **[CRITICAL]** The hint adds no new information. It simply rephrases the clues already present in the question. A valid hint must provide a *different* attribute from the text that was not mentioned in the question.
* **`ERR_LEAK` (Answer Disclosure)**: The hint contains the answer word, its root, or a direct synonym that makes the question trivial.
* **`ERR_TEXT_REF` (Structural Pointer)**: The hint references the physical layout of the text or the act of reading (e.g., "Look at the first paragraph," "The author says," or "The last sentence").
* **`ERR_HALLUC` (Outside Knowledge)**: The hint uses facts, names, or dates not found in the provided Source Passage, even if they are historically true in reality.

### RATING RUBRIC

* **Rating A (Excellent):** The hint provides a **semantic bridge** or logical nudge. It uses synonyms, paraphrasing, or conceptual clues to guide the student toward the answer without revealing it. It is pedagogically helpful and written in natural, fluent Farsi.
* **Rating B (Acceptable but Weak):** The hint is technically accurate but lacks depth. It may be too broad (`ERR_VAGUE`) or have minor stylistic/formatting issues (`ERR_FORMAT`). It doesn't fail the task, but it provides minimal guidance.
* **Rating C (Critical Failure):** The hint is unusable or counter-productive. It fails if it:
* **Leaks** the answer or a derivative of it (`ERR_LEAK`).
* **Directs** the user to a location like "sentence two" or "above" (`ERR_TEXT_REF`).
* **Hallucinates** facts not present in the Source Passage (`ERR_HALLUC`).
* **Repeats** the question without adding new information (`ERR_CIRC`).


OUTPUT FORMAT (Strict JSON):
{
  "Rating": "A/B/C",
  "Error_Class": "None/ERR_VAGUE/ERR_TEXT_REF/ERR_LEAK/ERR_HALLUC/ERR_CIRC",
  "Reasoning": "Explanation in English"
}

FEW-SHOT EXAMPLES:
---
**EXAMPLE 1 (Rating C - Textual Reference)**
Passage: "تخت جمشید نام یکی از شهرهای باستانی ایران است که سالیان سال پایتخت تشریفاتی پادشاهی ایران در زمان امپراتوری هخامنشیان بوده‌است. در این شهر باستانی کاخی به نام تخت جمشید وجود دارد که در دوران زمامداری داریوش بزرگ، خشایارشا و اردشیر اول بنا شده‌است و به مدت حدود ۲۰۰ سال آباد بوده‌است."
Question: "تخت جمشید پایتخت کدام امپراتوری بود؟"
Answer: "هخامنشیان"
Hint: "جواب در همان جمله اول متن ذکر شده است، لطفاً دوباره بخوانید."
Output: {"Rating": "C", "Error_Class": "ERR_TEXT_REF", "Reasoning": "The hint refers to 'sentence one' (جمله اول), which is a prohibited textual reference."}

**EXAMPLE 2 (Rating C - Hallucination)**
Passage: "سعدی شیرازی شاعر و نویسندهٔ پارسی‌گوی نامدار ایرانی است. آوازهٔ او بیشتر به خاطر نظم و نثر آهنگین، گیرا و قوی اوست. جایگاهش نزد اهل ادب تا بدان‌جاست که به وی لقب استاد سخن و شیخ اجل داده‌اند. آثار معروف او شامل گلستان و بوستان است."
Question: "آثار معروف سعدی کدامند؟"
Answer: "گلستان و بوستان"
Hint: "به دو کتابی فکر کنید که سعدی پس از بازگشت از سفرهای طولانی‌اش نوشت."
Output: {"Rating": "C", "Error_Class": "ERR_HALLUC", "Reasoning": "The text lists the books, but it never mentions that he wrote them 'after returning from long travels'. This implies external biographical knowledge."}

**EXAMPLE 3 (Rating A - Excellent)**
Passage: "یوزپلنگ آسیایی یا یوزپلنگ ایرانی یک زیرگونهٔ به‌شدت در معرض خطر انقراض از یوزپلنگ است که اکنون تنها چند ده قلاده از آن در ایران یافت می‌شود. این جانور سریع‌ترین حیوان خشکی‌زی است و در گذشته در مناطق وسیعی از آسیا زندگی می‌کرد."
Question: "وضعیت بقای یوزپلنگ آسیایی چگونه است؟"
Answer: "به‌شدت در معرض خطر انقراض"
Hint: "به تعداد بسیار کم این حیوان (تنها چند ده قلاده) که طبق گفته‌ها در ایران باقی مانده است توجه کنید."
Output: {
  "Rating": "A", 
  "Error_Class": "None", 
  "Reasoning": "The hint uses the specific population detail ('few dozen left') found in the text to trigger the concept of endangerment without explicitly pointing to 'the text' or 'sentences'."
}
---
Now evaluate the following input.
""",

    # --- 6. HINT REFINER ---
 "hint_refiner": """
You are an expert Educational Tutor (Farsi). A previous Hint was rejected.
Your task is to REGENERATE the hint based on the critic's reasoning to strictly satisfy the requirements.

Input Context:
1. **Passage:** The original text.
2. **QA Pair:** The question and answer.
3. **Failed Hint:** The hint that was rejected.
4. **Critic's Reasoning:** The specific reason for failure.

### CRITICAL RULES (Closed-Book / Faghat Matn):
1.  **NO HALLUCINATIONS (فقط متن):** You must **ONLY** use information explicitly written in the passage. Do not use outside knowledge or historical facts not present in the text.
2.  **NO CIRCULARITY:** Do not rephrase the question. Find a **different** attribute in the text.
3.  **NO TEXTUAL POINTERS:** Banned phrases: "در متن..." (In the text), "جمله اول..." (First sentence), "نویسنده می‌گوید..." (The author says).
4.  **NO LEAKAGE:** Do not use the answer word.

### REPAIR STRATEGIES:
* **IF ERROR = Circularity / Hallucination:**
   - *Issue:* You repeated the question or invented a fact.
   - *Fix:* Scan the text for a **different specific detail** (a name, a date, or an event) that is **explicitly** in the text and use that as the memory trigger.
* **IF ERROR = Textual Reference (ارجاع به متن):**
   - *Fix:* Identify *what* is in that sentence (e.g., a year) and describe it conceptually.

OUTPUT FORMAT (Strict JSON):
{
  "Reasoning": "Explain in English how you are fixing the issue (e.g., 'Using an explicit text detail I missed before').",
  "Hint": "The corrected, text-independent hint in Farsi."
}
""" , 

    "qa_hint_solver": """
You are a Content Strategist specializing in Educational Datasets.
We are trying to generate a "Hint" for a Question-Answer pair based on a Text, but the previous attempts FAILED.

### THE PROBLEM
The previous QA pair was likely "Un-Hintable." The attached log shows that the generated hints were repeatedly rejected by the Critic.

### YOUR TASK
1. **Analyze the Failures:** Look at the "Failed Hints & Reasons." Did they fail because they were Circular (repeating the question)? Or Hallucinations (making things up)?
   - If they were **Circular**, it means the Question already contains all the text's clues.
   - If they were **Hallucinations**, it means the Answer is too isolated in the text (no other facts to use).
2. **Select a NEW Answer:** Find a concept in the text that is **Context-Rich** (described by multiple distinct attributes/sentences).
3. **Generate a NEW Question:** Use one attribute for the Question.
4. **Reserve an Attribute:** Ensure there is at least one unused attribute left in the text that can serve as a Hint later.

### INPUT DATA
Text: {text}
Failed Question: {failed_q}
Failed Answer: {failed_a}
Failed Hints & Reasons: {failed_hints_list}

### OUTPUT FORMAT (Strict JSON)
{
  "Diagnosis": "I see the previous hints failed because [Reason]. This indicates the old QA pair was...",
  "Strategy": "I selected [New Answer] because it has two distinct details: [Detail 1] and [Detail 2].",
  "question": "The new Question based on Detail 1...",
  "answer": "The new Answer"
}
"""
}

# Master Dictionary
SYSTEM_PROMPTS = {
    "English": PROMPTS_EN,
    "Italian": PROMPTS_IT,
    "Farsi": PROMPTS_FA
}