#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow_datasets as tfds
import tensorflow as tf


# In[2]:


dataset, info =tfds.load('imdb_reviews/subwords8k',with_info=True,as_supervised=True)
train_dataset, test_dataset = dataset['train'],dataset['test']
encoder = info.features['text'].encoder
                        
                


# In[3]:


BUFFER_SIZE= 10000
BATCH_SIZE =64
padded_shapes=([None],())


# In[4]:


train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,padded_shapes=padded_shapes)
test_dataset = test_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,padded_shapes=padded_shapes)


# In[5]:


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=1)


# In[6]:


model=tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size,64),
                          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences=True)),
                          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)), 
                          tf.keras.layers.Dense(64,activation='relu'),
                          tf.keras.layers.Dropout(0.5), 
                          tf.keras.layers.Dense(1,activation='sigmoid')])
model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy'])
history = model.fit(train_dataset,epochs=3,validation_data=test_dataset,validation_steps=30)


# In[57]:


'''model=tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size,64),
                          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                          tf.keras.layers.Dense(64,activation='relu'),
                
                           tf.keras.layers.Dense(1,activation='sigmoid')])
model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy'])
history = model.fit(train_dataset,epochs=3,validation_data=test_dataset,validation_steps=30)'''


# In[7]:


model.save('prediction_model.h5')
def pad_to_size(vec,size):
    zeros =[0]*(size-len(vec))
    vec.extend(zeros)
    return vec


# In[8]:


def sample_predict(sentence, pad):
        encoded_sample_pred_text = encoder.encode(sentence)
        if pad:
            encoded_sample_pred_text =pad_to_size(encoded_sample_pred_text,64)
        encoded_sample_pred_text=tf.cast(encoded_sample_pred_text,tf.float32)
        predictions=model.predict(tf.expand_dims(encoded_sample_pred_text,0))
        
        return predictions
    


# In[15]:


sample_text=("What more can you ask for? A great screenplay based on one of the finest plays of the latter half of the 20th century, two fine emotional performances by Courtney and Finney, a realistic vision of war time london, a great supporting cast. This film takes you on an emotional rollercoaster through humour, sadness, loss and fulfillment. if you are in the theatre it is even more effective. This is a true 10 on the rating scale !" )
predictions =sample_predict(sample_text,pad=True)*100


# In[16]:


print (" %.2f" %predictions)


# In[ ]:





# In[ ]:




