# “Bangla Sentence Error Correction Tool”

This repository contains code for my “Bangla Sentence Error Correction” nlp tool using Seq2Seq model. The base project idea was an inspection of [this research paper](https://ieeexplore.ieee.org/abstract/document/8631974) [1] The idea was to implement a seq2seq model with LSTM encoder - decoder with attention.  

# Why this approach 
I opted for a sequence-to-sequence (seq2seq) model architecture due to its inherent advantages: end-to-end learning, contextual awareness, generalizability, and the ability to handle diverse error types. While I considered alternative approaches, rule-based systems, as discussed in [paper](https://ieeexplore.ieee.org/abstract/document/8554502) [2], require significant manual effort and domain expertise. Traditional machine learning approaches also face similar challenges, particularly in feature engineering and maintaining sequential context.

## Dataset 
I used this [dataset](https://github.com/hishab-nlp/BNSECData)
two files for train and test set as.csv format   
A small sample of the dataset looks like the  following 
| Target Input                                        | Target Output                                        |
|-----------------------------------------------------|------------------------------------------------------|
| পৃথিবীতে প্রতিটি মানুষই সুখী হতে চায়, কিন্তু সুখ কী এবং তা কীভাবে পাওয়া যায় তা অনেকেই জানেন না। | পৃথিবীতে প্রতিটি মানুষই সুখি হতে চায়, কিন্তু সুখি কী এবং তা কীভাবে পাওয়া যায় তা অনেকেই জানে না। |
| তোমার মনে সুত আছে তাই তুমি এখন সুখী।         | তোমার মনে সুত আছে তাই তুমি এখন সুখী।            |
| আমাদের জীবনকে সুন্দর করতে প্রকৃতির রয়েছে অসীম ভূমিকা।       | আমাদের জীবনের সুন্দর করতে প্রকৃতির রয়েছে অসীম ভূমিকা।         |
| পৃথিবীর সবজেয়ে বড় গ্রন্থাগারটি হল কংগ্রেস গ্রন্থাগার যা যুক্তরাষ্ট্রে রাজধানী ওয়াশিংটন ডিসিতে অবস্থিত। | পৃথিবীর সবজেয়ে বড় গ্রন্থাগারটি হল কংগ্রেস গ্রন্থাগার যা যুক্তরাষ্ট্রের রাজধানী ওয়াশিংটন ডিসিতে অবস্থিত। |
| এই কমর রগমে ঠাণ্ডা পানি পান করার মজাই আলাদা।  | এই রকম গরমে ঠাণ্ডা পানি পান করার মজাই আলাদা।      |
| সূর্যের আলো আমাদের জন্য খুবই উপকারী, তবে অতিরিক্ত সূর্যের আলো আমাদের ত্বকের জন্য ক্ষতিকর।   | সূর্যের আলো আমাদের জন্য খুবই উপকারী, তবে অতিরিক্ত সূর্যের আলো আমাদের ত্বকের জন্য ক্ষতিকারক। |
| বাংলাদেশে প্রতি বছর প্রায় ১০ লাখ টন চাউল উৎপাদন হয়।   | বাংলাদেশে প্রতি বছর প্রায় ১০ লাখ টন চাল উৎপাদন হয়।         |
| বাংলাদেশে প্রতি বছর প্রায় ১০ লাখ টন চাল উৎপাতন হয়।   | বাংলাদেশে প্রতি বছর প্রায় ১০ লাখ টন চাল উৎপাতন হয়।         |

There are 1356300 samples is train set and 900 samples in test set 

 __***Important note: With the resources  I had at hand (google colab free version), if I we're to fit the training data I would have to train each epoch for 38:40:42 hours on average and even on the smaller fractions of it I couldn't make it past the 3 hour limit. So I had to settle for 0.04% of the data (54252 samples) which would barely let me fit the data in colab gpu and finish the training ***__  


# Training information 

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- nltk

## Installation

```bash
pip install tensorflow pandas numpy scikit-learn nltk
```
## General training information 
1.  **Data Preprocessing**:
    
    -   Load the training and testing data from CSV files.
    -   Perform data cleaning by calling `drop_null_duplicate()`
 function
     -   Preprocess the text data by removing punctuations, emojis, and other characters by applying `preprocessing(text)` to the columns .
2.  **Tokenization and Padding**:
    
    -   Tokenized the input and target text using the `Tokenizer` class from Keras.
    -   Paded the tokenized sequences to ensure uniform length.
3.  **Model Architecture**:
    
    -   Defined the encoder-decoder architecture with attention mechanism.
    -   Utilized LSTM layers for sequence processing.
    -   Used an embedding layer for word embeddings.
4.  **Model Training**:
    
    -   Compiled the model with the Adam optimizer and sparse categorical cross-entropy loss function.
    -   Trained the model on the preprocessed data with a  epochs of 10 and batch size 16.
# Evaluation discussion and shortcomings 
The model performs relatively well but there are areas to improve. Lets discuss about them

## Evaluation  Score: 
The evaluation scores are poor.  The WER  CER and BLEU scores are: 
| Metric       | Value                  |
|--------------|------------------------|
| Average WER  | 0.46909027191250546    |
| Average CER  | 0.07864318477783686    |
| BLEU Score   | 0.8366168205319402     |
## Discussion:  

The low WER and CER values suggest that the model's word and character predictions are highly accurate, with the majority of words and characters being correctly predicted. The high BLEU score further corroborates the strong performance of the model, indicating a high degree of similarity between the predicted and reference outputs. The model could have worked better but it didn't. The primary reason is the fraction of dataset I took was extremely  small compared to the entirety of the train set. So the model caught into the random noises and also failed to generalize. Another thing is the size of the tokenizer was around 17k. As there was less variation of words the model could learn and adapt to the newer problems.   
## Proposed Improvements:
I intend to explore some solutions .They solutions are 

 1.  I intend to explore and implement the problem again with
    TensorFlow dataloader to create a input pipeline to handle the large
    data issue
 2. I don't advocate for splitting the dataset into smaller parts and
    training the model on each part because it will affect the model
    performance in the long run

3. I want to explore using BanglaBERT model and try finetuning the model more 
4. The biggest issue I have yet to solve is the colab and kaggle cooldown time. I will retry the solution again with checkpoints   





# Reference  and resources 

## References:
 [1]S. Islam, M. F. Sarkar, T. Hussain, M. M. Hasan, D. M. Farid and S. Shatabda, "Bangla Sentence Correction Using Deep Neural Network Based Sequence to Sequence Learning," _2018 21st International Conference of Computer and Information Technology (ICCIT)_, Dhaka, Bangladesh, 2018, pp. 1-6, doi: 10.1109/ICCITECHN.2018.8631974.  
 [2]M. Mashod Rana, M. Tipu Sultan, M. F. Mridha, M. Eyaseen Arafat Khan, M. Masud Ahmed and M. Abdul Hamid, "Detection and Correction of Real-Word Errors in Bangla Language," _2018 International Conference on Bangla Speech and Language Processing (ICBSLP)_, Sylhet, Bangladesh, 2018, pp. 1-4, doi: 10.1109/ICBSLP.2018.8554502.  


## Resources I took help from :

 1. Sequence-to-Sequence (seq2seq) Encoder-Decoder Neural Networks, Clearly Explained!!! [\[Link\]](https://www.youtube.com/watch?v=L8HKweZIOmg)
 2. টেন্সরফ্লো দিয়ে সহজ বাংলায় 'বাংলা' ন্যাচারাল ল্যাঙ্গুয়েজ প্রসেসিং [\[Link\]](https://github.com/raqueeb/nlp_bangla/tree/master)
 3. A ten-minute introduction to sequence-to-sequence learning in Keras[\[Link\]](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)
 4. Anomaly Detection in NLP Using Levenshtein Distance[\[Link\]](https://medium.com/munchy-bytes/anomaly-detection-in-nlp-using-levenshtein-distance-62351639d189)
