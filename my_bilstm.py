import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import math
import pickle
import logging
import argparse
import itertools
import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
from sklearn.metrics import classification_report
import nltk
nltk.download('punkt')

class TFNer(tf.keras.Model):
    def __init__(self, max_seq_len, embed_input_dim, embed_output_dim, num_labels, weights):
        super(TFNer, self).__init__()
        self.embedding = layers.Embedding(input_dim=embed_input_dim, output_dim=embed_output_dim, weights=weights, input_length=max_seq_len, trainable=False, mask_zero=True)
        self.bilstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))
        self.dense = layers.Dense(num_labels)

    def call(self, inputs):
        x = self.embedding(inputs) # batchsize, max_seq_len, embedding_output_dim
        x = self.bilstm(x) #batchsize, max_seq_len, hidden_dim_bilstm
        logits = self.dense(x) #batchsize, max_seq_len, num_labels
        return logits

def split_text_label(filename):
    '''
    Reads a file named filename, extracts the text and the labels and stores
    them in an array.
     
    returns [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ] 

    '''
    f = open(filename)
    split_labeled_text = []
    sentence = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                split_labeled_text.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        sentence.append([splits[0],splits[-1].rstrip("\n")])
    if len(sentence) > 0:
        split_labeled_text.append(sentence)
        sentence = []
    return split_labeled_text

def padding(sentences, labels, max_len, padding='post'):
    padded_sentences = pad_sequences(sentences, max_len, padding='post')
    padded_labels = pad_sequences(labels, max_len, padding='post')
    return padded_sentences, padded_labels

def createMatrices(data, word2Idx, label2Idx):
    sentences = []
    labels = []
    for split_labeled_text in data:
        wordIndices = []
        labelIndices = []
        for word, label in split_labeled_text:
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]                 
            else:                
                wordIdx = word2Idx['UNKNOWN_TOKEN']
            wordIndices.append(wordIdx)
            labelIndices.append(label2Idx[label])
    
        sentences.append(wordIndices)
        labels.append(labelIndices)
    return sentences, labels

def idx_to_label(predictions, correct, idx2Label): 
    label_pred = []    
    for sentence in predictions:
        for i in sentence:
            label_pred.append([idx2Label[elem] for elem in i ]) 

    label_correct = []  
    if correct != None:
        for sentence in correct:
            for i in sentence:
                label_correct.append([idx2Label[elem] for elem in i ]) 
        
    return label_correct, label_pred

def main():

    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s', datefmt='%m/%d/%Y ', level=logging.INFO)
    logger = logging.getLogger(__name__)


    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data", default=None, type=str, required=True,help="Directory which has the data files for the task")
    parser.add_argument("--output", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--overwrite", default=False, type=bool, help="Set it to True to overwrite output directory")


    args = parser.parse_args()

    if os.path.exists(args.output) and os.listdir(args.output) and not args.overwrite:
        raise ValueError("Output directory ({}) already exists and is not empty. Set the overwrite flag to overwrite".format(args.output))
    if not os.path.exists(args.output):
        os.makedirs(args.output)
   

    train_batch_size = 32
    valid_batch_size = 64
    test_batch_size = 64

    # padding sentences and labels to max_length of 128
    max_seq_len = 128
    EMBEDDING_DIM = 100
    epochs = 30

    split_train = split_text_label(os.path.join(args.data, "train.txt"))
    split_valid = split_text_label(os.path.join(args.data, "valid.txt"))
    split_test = split_text_label(os.path.join(args.data, "test.txt"))

    test_words = []
    test_true_labels = []
    for sentence in split_test:
        for word in sentence:
            test_words.append(word[0])
            test_true_labels.append(word[1])

    test_words = " ".join(test_words)
    # print(test_true_labels, len(test_words), len(test_true_labels))
    # print(split_test)

    labelSet = set()
    wordSet = set()
    # words and labels 
    for data in [split_train, split_valid, split_test]:
        for labeled_text in data:
            for word, label in labeled_text:
                labelSet.add(label)
                wordSet.add(word.lower())

    # Sort the set to ensure '0' is assigned to 0
    sorted_labels = sorted(list(labelSet), key=len)

    # Create mapping for labels
    label2Idx = {}
    for label in sorted_labels:
        label2Idx[label] = len(label2Idx)
    num_labels = len(label2Idx)
    idx2Label = {v: k for k, v in label2Idx.items()}
    
    pickle.dump(idx2Label,open(os.path.join(args.output, "idx2Label.pkl"), 'wb'))
    logger.info("Saved idx2Label pickle file")

    # Create mapping for words 
    word2Idx = {}
    if len(word2Idx) == 0:
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
    for word in wordSet:
        word2Idx[word] = len(word2Idx)
    logger.info("Total number of words is : %d ", len(word2Idx))
    pickle.dump(word2Idx, open(os.path.join(args.output, "word2Idx.pkl"), 'wb'))
    logger.info("Saved word2Idx pickle file")

    # Loading glove embeddings 
    embeddings_index = {}
    f = open('embeddings/glove.6B.100d.txt', encoding="utf-8")
    for line in f:
        values = line.strip().split(' ')
        word = values[0] # the first entry is the word
        coefs = np.asarray(values[1:], dtype='float32') #100d vectors representing the word
        embeddings_index[word] = coefs
    f.close()
    logger.info("Glove data loaded")

    #print(str(dict(itertools.islice(embeddings_index.items(), 2))))

    embedding_matrix = np.zeros((len(word2Idx), EMBEDDING_DIM))
    
    # Word embeddings for the tokens    
    for word,i in word2Idx.items():
        embedding_vector = embeddings_index.get(word)      
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    pickle.dump(embedding_matrix, open(os.path.join(args.output, "embedding.pkl"), 'wb'))
    logger.info("Saved Embedding matrix pickle")
  
    # Interesting - to check how many words were not there in Glove Embedding
    # indices = np.where(np.all(np.isclose(embedding_matrix, 0), axis=1))
    # print(len(indices[0]))
    train_sentences, train_labels = createMatrices(split_train, word2Idx, label2Idx)
    valid_sentences, valid_labels = createMatrices(split_valid, word2Idx, label2Idx)
    test_sentences, test_labels = createMatrices(split_test, word2Idx, label2Idx)

    
    train_features, train_labels = padding(train_sentences, train_labels, max_seq_len, padding='post' )
    valid_features, valid_labels = padding(valid_sentences, valid_labels, max_seq_len, padding='post' )
    test_features, test_labels = padding(test_sentences, test_labels, max_seq_len, padding='post' )


    logger.info(f"Train features shape is {train_features.shape} and labels shape is{train_labels.shape}")
    logger.info(f"Valid features shape is {valid_features.shape} and labels shape is{valid_labels.shape}")
    logger.info(f"Test features shape is {test_features.shape} and labels shape is{test_labels.shape}")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_features, valid_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))


    shuffled_train_dataset = train_dataset.shuffle(buffer_size=train_features.shape[0], reshuffle_each_iteration=True)

    batched_train_dataset = shuffled_train_dataset.batch(train_batch_size, drop_remainder=True)
    batched_valid_dataset = valid_dataset.batch(valid_batch_size, drop_remainder=True)
    batched_test_dataset = test_dataset.batch(test_batch_size, drop_remainder=True)

    epoch_bar = master_bar(range(epochs))
    train_pb_max_len = math.ceil(float(len(train_features))/float(train_batch_size))
    valid_pb_max_len = math.ceil(float(len(valid_features))/float(valid_batch_size))
    test_pb_max_len = math.ceil(float(len(test_features))/float(test_batch_size))

    model = TFNer(max_seq_len=max_seq_len, embed_input_dim=len(word2Idx), embed_output_dim=EMBEDDING_DIM, weights=[embedding_matrix], num_labels=num_labels)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_log_dir = f"{args.output}/logs/train"
    valid_log_dir = f"{args.output}/logs/valid"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    
    train_loss_metric = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
    valid_loss_metric = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)

    def train_step_fn(sentences_batch, labels_batch):
        with tf.GradientTape() as tape:
            logits = model(sentences_batch) # batchsize, max_seq_len, num_labels
            loss = scce(labels_batch, logits) #batchsize,max_seq_len
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
        return loss, logits

    def valid_step_fn(sentences_batch, labels_batch):
        logits = model(sentences_batch)
        loss = scce(labels_batch, logits)
        return loss, logits
    
    for epoch in epoch_bar:
        with train_summary_writer.as_default():
            for sentences_batch, labels_batch in progress_bar(batched_train_dataset, total=train_pb_max_len, parent=epoch_bar) :
                
                loss, logits = train_step_fn(sentences_batch, labels_batch)
                train_loss_metric(loss)
                epoch_bar.child.comment = f'training loss : {train_loss_metric.result()}'
            tf.summary.scalar('training loss', train_loss_metric.result(), step=epoch)
            train_loss_metric.reset_states()

        with valid_summary_writer.as_default():
            for sentences_batch, labels_batch in progress_bar(batched_valid_dataset, total=valid_pb_max_len, parent=epoch_bar):
                loss, logits = valid_step_fn(sentences_batch, labels_batch)
                valid_loss_metric.update_state(loss)
                
                epoch_bar.child.comment = f'validation loss : {valid_loss_metric.result()}'
          
            # Logging after each Epoch !
            tf.summary.scalar('valid loss', valid_loss_metric.result(), step=epoch)
            valid_loss_metric.reset_states()

    model.save_weights(f"{args.output}/model_weights",save_format='tf')  
    logger.info(f"Model weights saved")
   
   
    #Evaluating on test dataset 

    test_model =  TFNer(max_seq_len=max_seq_len, embed_input_dim=len(word2Idx), embed_output_dim=EMBEDDING_DIM, weights=[embedding_matrix], num_labels=num_labels)
    test_model.load_weights(f"{args.output}/model_weights")
    logger.info(f"Model weights restored")

    # true_labels = []
    # pred_labels = []

    # for sentences_batch, labels_batch in progress_bar(batched_test_dataset, total=test_pb_max_len):
        
    #     logits = test_model(sentences_batch)
    #     temp1 = tf.nn.softmax(logits)       
    #     preds = tf.argmax(temp1, axis=2)
    #     true_labels.append(np.asarray(labels_batch))
    #     pred_labels.append(np.asarray(preds))
    # label_correct, label_pred = idx_to_label(pred_labels, true_labels, idx2Label)
    # report = classification_report(label_correct, label_pred)
    # logger.info(f"Results for the test dataset")
    # logger.info(f"\n{report}")

# ------------------------------------test--------------------------------
    # test_words = 'I live in Hong Kong and Anna was born in China'
    test_words = list(test_words.split(" "))
    total_length = len(test_words)
    total_labels = []
    count = int(len(test_words)/128)
    for i in range(count+1):
        if i == count:
            sentence = test_words[i*128:]
            for i in range(total_length - i * 128):
                sentence.append('None')
        else:
            sentence = test_words[i*128:(1+i)*128]
        sentences = []
        wordIndices = []
        masks = []
        length = len(sentence)

        for word in sentence:
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]                 
            else:                
                wordIdx = word2Idx['UNKNOWN_TOKEN']
            wordIndices.append(wordIdx)
        maskindices = [1]*len(wordIndices)
        sentences.append(wordIndices)
        masks.append(maskindices)
        padded_inputs = pad_sequences(sentences, maxlen=max_seq_len, padding="post")
        masks = pad_sequences(masks, maxlen=max_seq_len, padding="post")

        padded_inputs = tf.expand_dims(padded_inputs, 0)
        true_labels = None
        pred_labels = []
        pred_logits = []
        for sentence in padded_inputs:
            logits = test_model(sentence)
            temp1 = tf.nn.softmax(logits) 
            max_values = tf.reduce_max(temp1,axis=-1)
            masked_max_values = max_values * masks 
            preds = tf.argmax(temp1, axis=2)
            pred_labels.append(np.asarray(preds))
            pred_logits.extend(np.asarray(masked_max_values))
        _,label_pred  = idx_to_label(pred_labels, true_labels,idx2Label)
        label_pred = label_pred[0][:length] 
        # print('111111',label_pred, len(label_pred))
        total_labels.append(label_pred)
        # pred_logits = pred_logits[0][:length]
        # words = word_tokenize(test_words)
        # assert len(label_pred) == len(words)
        # zip_val = zip(words, label_pred)
        
        # output = [{"word":word,"tag":label} for  word, label in zip_val]
        # print(output, len(output(attrs=None, header='Set-Cookie:')))
    # print('22222222222222222222222222', len(np.array(total_labels).reshape(-1)))
    pred_labels = np.array(total_labels).reshape(-1)[:total_length]
    # print(len(test_words))
    # print(len(pred_labels))
    # print(len(test_true_labels))
    f = open('result.txt', 'a')
    for i in range(total_length):
        f.write(test_words[i])
        f.write(' ')
        f.write(pred_labels[i])
        f.write(' ')
        f.write(test_true_labels[i])
        f.write('\n')
    f.close()



if __name__ == "__main__":
    main()
   
    