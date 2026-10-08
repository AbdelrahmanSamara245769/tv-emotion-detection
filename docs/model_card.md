#  Model Card – RoBERTa-Large (Class-Weighted Loss)

###  Project Information
**Project:** Emotion Detection in TV Transcripts  
**Model:** RoBERTa-Large (Class-Weighted Loss)  
**Trained on:** Breda University of Applied Sciences GPU Server  

---

##  Model Overview
Our final model is **RoBERTa-Large with class-weighted loss**.  
It predicts seven emotions — *happiness, sadness, anger, surprise, fear, disgust,* and *neutral.*  
This version performed best in all tests, with higher balance across emotion types.

---

##  Architecture
- **Base model:** RoBERTa-Large (24 Transformer layers, 355M parameters)  
- **Loss:** Class-weighted cross-entropy (weights ≈ 0.7 – 2.5)  
- **Optimizer:** AdamW  
- **Learning rate:** 1e-5  
- **Batch size:** 32  
- **Epochs:** 4  
- **Max length:** 256  
- **Warm-up ratio:** 0.06  
- **Weight decay:** 0.01  
- **Scheduler:** Cosine learning rate  
- **Seed:** 42  
- **Hardware:** NVIDIA T4 GPU (Breda University Server)  
- **Training time:** ~25 minutes (4 epochs) 


---

##  Purpose
This model was created to detect emotions in **television show transcripts**.  
It helps the **Content Intelligence Agency** automatically identify emotional tone in dialogue.  
The goal is to improve media analysis, episode tagging, and content understanding.  

---

##  Development Context
- **Dataset:** English transcripts from TV shows  
- **Assumptions:** Each sentence expresses one main emotion  
- **Limitations:**  
  - Imbalanced dataset (few *disgust* and *surprise* examples)  
  - Text-only context (no audio or visual input)  
- **Training resources:** Breda University GPU server (balanced compute access)

---

##  Intended Use
**Use for:**  
- Detecting emotions in subtitles or transcribed dialogue  
- Media research or automatic tagging for video archives  

**Avoid using for:**  
- Psychological or mental health assessment  
- Real-time emotion detection in live speech  
- Non-English data without fine-tuning  

---

##  Dataset Details

### Dataset Origin and Selection (Added for Clarity)
We tested two different datasets before choosing the final one used to train this model.

- #### Dataset 1

About 73,000 sentences
Taken from TV show dialogue (e.g. Friends)
Sentences were long and conversational
Often contained multiple emotions in one sentence
Many filler or neutral words
Because of this, the model had trouble learning the main emotion of each sentence, and performance was lower.

- #### Dataset 2 (Final Dataset Used)

About 26,000 sentences
Short, clear sentences
Each sentence shows one main emotion (happy, sad, angry, etc.)
Text comes from TV show transcripts, including emotional dialogue similar to Australian Survivor
This dataset was easier for the model to learn from and gave much better results, so we decided to use Dataset 2 only for all final experiments.

- #### How the Data Was Used
  
The 26,000 sentences were cleaned
Removed duplicates and missing values
Fixed encoding issues
Removed URLs and special characters
Standardized emotion labels (e.g. joy → happiness)

- #### The data was split into:

80% training
20% validation
A separate test set of ~600 sentences was provided by the client and used only for final evaluation

- #### Dataset Limitations

Some emotions (disgust and surprise) appear very rarely
This makes them harder for the model to predict
Sentences with mixed or weak emotional wording can confuse the model
To reduce this issue, we used class-weighted loss so that rare emotions had more influence during training. 

---

##  Performance Metrics and Evaluation
| Metric | Score |
|--------|--------|
| **Accuracy** | **0.693** |
| **Macro F1** | **0.632** |
| **Weighted F1** | **0.692** |
| **Macro Precision** | 0.633 |
| **Macro Recall** | 0.645 |

**Summary:**  
- Best overall model performance.  
- Improved prediction for minority emotions (*surprise, anger*).  
- Weakest on *disgust* due to very few samples.  

 **Error Analysis:** 
 
Summary of Error Analysis: 

We performed a detailed error analysis on our Weighted RoBERTa-Large model.  
Although the model achieved a weighted F1 of 0.692 and accuracy 0.693, its macro F1 (0.632) showed that rare emotions like disgust and surprise were harder to classify.  
Confusion matrices revealed overlaps between *neutral–happiness* and *sadness–fear*, while longer or mixed-tone sentences increased misclassifications.  
These findings explain where the model struggles and guided our improvements with class weighting and data balancing.
- [Error analysis notebook](../notebooks/11_error_analysis.ipynb)

---

##  Explainability and Transparency (XAI)
We applied three complementary methods to interpret our RoBERTa-Large emotion classification model. First, we used Gradient × Input to identify which words most influenced each prediction. Then, we applied Layer-wise Relevance Propagation (LRP) with conservative propagation to improve interpretability and focus on emotionally meaningful tokens. Finally, we performed input perturbation, one by one removing tokens to test how the model confidence changes. Together, these methods allowed us to understand how the model processes emotional language and evaluate its robustness and focus.

- [Explainability notebook](../notebooks/12_explainability.ipynb)
---

##  Recommendations for Use
- Recommended model for text-based **emotion classification**.  
- Works best on conversational or script-style data.  
- For faster inference, consider **DistilRoBERTa** with similar training setup.  
- Re-train if applied to other domains (e.g., social media).  

---

##  Sustainability Considerations
- **Hardware:** NVIDIA T4 GPU  
- **Training time:** ~ 25 minutes 
- **Energy use:** ≈ Estimated 0.8 kWh  
- **Carbon impact:** Low  
- **Tips for lower footprint:**  
  - Use pre-trained weights instead of retraining from scratch  
  - Batch inference to reduce compute waste  
  - Use cloud providers with renewable energy  

---

##  Summary
The **RoBERTa-Large (Class-Weighted Loss)** model is the best-performing and most balanced model for this project.  
It achieved **0.693 accuracy** and **0.632 macro-F1**, improving emotion detection for most classes.  
It is the **recommended final model** for deployment unless data constraints change.

---

