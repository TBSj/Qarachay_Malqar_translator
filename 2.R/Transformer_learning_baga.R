# 1. Training                                         ####
#    1.0. Libraries                                   ####
library(tensorflow)
library(keras)
library(tfdatasets)
library(this.path)
library(purrr)
library(stringr)
library(reshape2)
library(viridis)
library(ggplot2)
library(tibble)
library(plotly)
library(data.table)
library(reticulate)
library(tokenizers.bpe)
# library(R6)
# reticulate::py_install('nltk', pip = TRUE)
nltk = reticulate::import('nltk')


FILEPATH_SOURCE_PREPARED <- str_replace(this.dir(), strsplit(this.dir(), '/')[[1]] %>% last(), "Source_prepared")

FILE_TEXT <- file.path(FILEPATH_SOURCE_PREPARED, "Bary_bir_to_model.csv")
# FILE_TEXT <- file.path(FILEPATH_SOURCE_PREPARED, "Bary_besh_to_model.csv")

SRC_LANG <- 'rus'
TRG_LANG <- 'krc'

# SRC_LANG <- 'krc'
# TRG_LANG <- 'rus'

MODEL_PATH_SRC_BPE    <- file.path(this.dir(), 'BPE', paste0("youtokentome_", SRC_LANG, ".bpe"))
MODEL_PATH_TARGET_BPE <- file.path(this.dir(), 'BPE', paste0("youtokentome_", TRG_LANG, ".bpe"))

TRANSFORMER_NAME <- paste0('transformer_', SRC_LANG, '_to_',TRG_LANG, '_Baga')

TRANSFORMER_FILE <- file.path(this.dir(), 'model', TRANSFORMER_NAME, TRANSFORMER_NAME)

#    1.1. Text                                        ####
#         1.1.1. Download dataset                     ####
text_pairs <- FILE_TEXT %>%
  fread()
# .[seq(250)] # for examples

str(text_pairs[sample(nrow(text_pairs), 1), ])

# src_vocab_size_for_bpe    <- round(strsplit(text_pairs[[SRC_LANG]],    ' ') %>% unlist() %>% uniqueN() * 0.7)
# target_vocab_size_for_bpe <- round(strsplit(text_pairs[[TRG_LANG]], ' ') %>% unlist() %>% uniqueN() * 0.7)

#         1.1.2. BPE                                  ####
# model_src_bpe <- bpe(x      = text_pairs[[SRC_LANG]],
#                  coverage   = 0.999,
#                  vocab_size = src_vocab_size_for_bpe,
#                  threads    = 1,
#                  model_path = MODEL_PATH_SRC_BPE)

# model_trg_bpe <- bpe(x         = text_pairs[[TRG_LANG]],
#                      coverage   = 0.999,
#                      vocab_size = target_vocab_size_for_bpe,
#                      threads    = 1,
#                      model_path = MODEL_PATH_TARGET_BPE)

model_src_bpe <- bpe_load_model(MODEL_PATH_SRC_BPE)

model_trg_bpe <- bpe_load_model(MODEL_PATH_TARGET_BPE)



# bpe_decode(model_src, src_diglist) %>% unlist()

src_diglist <- bpe_encode(model = model_src_bpe, 
                          x     = text_pairs[[SRC_LANG]], 
                          type  = "ids", 
                          bos   = TRUE, 
                          eos   = TRUE)

src_maxlen <- lapply(src_diglist, length) %>% unlist() %>% max()



trg_diglist <- bpe_encode(model = model_trg_bpe, 
                          x     = text_pairs[[TRG_LANG]], 
                          type  = "ids", 
                          bos   = TRUE, 
                          eos   = TRUE)

trg_maxlen <- lapply(trg_diglist, length) %>% unlist() %>% max()

# sequence_length <- max(trg_maxlen, src_maxlen)

src_matrix <-
  pad_sequences(src_diglist, maxlen = src_maxlen,  padding = "post")

trg_matrix <-
  pad_sequences(trg_diglist, maxlen = trg_maxlen + 1, padding = "post")

#         1.1.3. Train-test-split                     ####
num_test_samples <- num_val_samples <- round(0.1 * nrow(text_pairs))

num_train_samples <- nrow(text_pairs) - num_val_samples - num_test_samples

pair_group <- sample(c(
  rep("train", num_train_samples),
  rep("test", num_test_samples),
  rep("val", num_val_samples)
))

# train_pairs <- text_pairs[pair_group == "train", ]
test_pairs <- text_pairs[pair_group == "test", ]
# val_pairs <- text_pairs[pair_group == "val", ]

x_train <- src_matrix[pair_group == "train",]
y_train <- trg_matrix[pair_group == "train",]

x_valid <- src_matrix[pair_group == "val",]
y_valid <- trg_matrix[pair_group == "val",]

#    1.2. Hyperparameters / variables                 ####
#         1.2.1. Hyperparameters / variables          ####
batch_size <- 64
# embedding_dim <- 64
# gru_units <- 256
# enc_dropout = 0.1
# dec_dropout = 0.1
# src_maxlen <- 9
# target_maxlen <- 15

src_vocab_size    <- model_src_bpe$vocab_size
target_vocab_size <- model_trg_bpe$vocab_size

num_layers <- 6 # Количество энкодер и декодер слоёв (6)
d_model <- 256 # Количество нейронов в слое (или глубина внимания) (Эм иги 512, компьютерим тарталмайды)
dff <- 1024 # Расстояние позиций (токенов) для слоя позиционно-зависимого прямого распространения (Эм иги 2048, компьютерим тарталмайды)
num_heads <- 4 # Кол-во голов внимания (8)
dropout_rate <- 0.3 # Дропаут
pe_input <- dff
pe_target <- dff

buffer_size <- nrow(x_train)

tfVal32 <- function(x){
  tf$cast(x, dtype=tf$int32)
}
tfVal64 <- function(x){
  tf$cast(x, dtype=tf$int64)
}
tfValFl32 <- function(x){
  tf$cast(x, dtype=tf$float32)
}
tfValFl64 <- function(x){
  tf$cast(x, dtype=tf$float64)
}

#         1.2.2. Create datasets                      ####
train_dataset <-
  tensor_slices_dataset(keras_array(list(x_train, y_train)))  %>%
  dataset_shuffle(buffer_size = buffer_size) %>%
  dataset_batch(batch_size, drop_remainder = TRUE)

validation_dataset <-
  tensor_slices_dataset(keras_array(list(x_valid, y_valid))) %>%
  dataset_shuffle(buffer_size = buffer_size) %>%
  dataset_batch(batch_size, drop_remainder = TRUE)

c(inputs, targets) %<-% iter_next(as_iterator(train_dataset))
c(inputs_v, targets_v) %<-% iter_next(as_iterator(validation_dataset))

#    1.3. Model                                       ####
#         1.3.1. Optimizer                            ####
# CustomSchedule <- R6Class(classname = 'CustomSchedule',
#                           # inherit = keras$optimizers$schedules$LearningRateSchedule,
#                           
#                           public = list(
#                             
#                             d_model = NULL,
#                             warmup_steps = NULL,
#                             
#                             initialize = function(d_model, warmup_steps=4000) {
#                               
#                               self$d_model <- tfValFl32(d_model)
#                               self$warmup_steps <- warmup_steps
#                               
#                               
#                             },
#                             
#                             call = function(step){
#                               arg1 <- tf$math$rsqrt(step)
#                               arg2 <- step * (self$warmup_steps ** -1.5)
#                               
#                               return(tf$math$rsqrt(self$d_model) * tf$math$minimum(arg1, arg2))
#                               
#                             }
#                             
#                           ),
#                           private = list(
#                           )
# )



CustomSchedule (tf$keras$optimizers$schedules$LearningRateSchedule) %py_class% {
  `__init__` <- function(d_model, warmup_steps=4000) {
    super()$`__init__`()
    self$d_model <- tfValFl32(d_model)
    self$warmup_steps <- warmup_steps
  }
  `__call__` <- function(step){
    arg1 <- tf$math$rsqrt(step)
    arg2 <- step * (self$warmup_steps ** -1.5)
    
    return(tf$math$rsqrt(self$d_model) * tf$math$minimum(arg1, arg2))
    
  }
}


learning_rate <- CustomSchedule(d_model)

optimizer <- tf$keras$optimizers$Adam(learning_rate,
                                      beta_1=0.9,
                                      beta_2=0.98,
                                      epsilon=1e-9)

# optimizer <- tf$keras$optimizers$Adam(0.001, 
#                                       beta_1=0.9, 
#                                       beta_2=0.98,
#                                       epsilon=1e-9)

#         1.3.2. Loss function                        ####   
loss_object <- tf$keras$losses$SparseCategoricalCrossentropy(
  from_logits=TRUE, reduction='none')

# cx_loss <- function(y_true, y_pred) {
#   mask <- ifelse(y_true == 0L, 0, 1)
#   # loss <-
#   #   tf$nn$sparse_softmax_cross_entropy_with_logits(labels = y_true,
#   #                                                  logits = y_pred) * mask
#   loss <-
#     tf$keras$losses$SparseCategoricalCrossentropy(labels = y_true,
#                                                   logits = y_pred) * mask
#   
#   return(tf$reduce_mean(loss) / tf$reduce_mean(mask))
# }

# loss_object <- tf$keras$losses$SparseCategoricalCrossentropy(
  # from_logits=TRUE, reduction='none')

loss_function <- function(real, pred){
  mask <- tf$math$logical_not(tf$math$equal(real, tf$cast(0, dtype = real$dtype)))
  
  loss_ <- loss_object(real, pred)
  
  mask <- tf$cast(mask, dtype=loss_$dtype)
  loss_ = loss_ * mask
  
  return(tf$reduce_sum(loss_)/tf$reduce_sum(mask))
}


#         1.3.3. Positional encoding                  ####
# Attachments represent the token in a d-dimensional space, where tokens with a similar value will be closer to each other. But attachments do not encode the relative position of the tokens in the sentence. Thus, after adding the positional encoding, the tokens will be closer to each other based on the similarity of their meaning and their position in the sentence, in d-dimensional space.

get_angles <- function(pos, i, d_model){
  angle_rates = 1 / (10000 ** ((2 * (i %/% 2)) / as.double(d_model)))
  return(pos %*% angle_rates)
}

positional_encoding <- function(position, d_model){
  angle_rads <- get_angles(pos = (seq(position)-1) %>% as.matrix(),
                           i = (seq(d_model)-1) %>% as.matrix() %>% t(),
                           d_model = d_model)
  # apply sin to even indices in the array; 2i
  angle_rads[, seq(1, dim(angle_rads)[2], by = 2)] <- sin(angle_rads[, seq(1, dim(angle_rads)[2], by = 2)])
  # apply cos to odd indices in the array; 2i+1
  angle_rads[, seq(2, dim(angle_rads)[2], by = 2)] <- cos(angle_rads[, seq(2, dim(angle_rads)[2], by = 2)])
  return(k_expand_dims(tfValFl32(angle_rads), axis = 1))
}




#         1.3.4. Mask                                 ####
create_padding_mask <- function(seq){
  seq <- tfValFl32(tf$math$equal(tfValFl32(seq), 0))
  return(seq[,tf$newaxis, tf$newaxis,])
}

create_look_ahead_mask <- function(size){
  mask <- 1 - tf$linalg$band_part(tf$ones(tfVal32(c(size, size))), tfVal32(-1), tfVal32(0))
}
#         1.3.5. Scaled dot-product attention         ####
scaled_dot_product_attention <- function(q, k, v, mask){
  # """Calculate the attention weights.
  # q, k, v must have matching leading dimensions.
  # k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  # The mask has different shapes depending on its type(padding or look ahead)
  # but it must be broadcastable for addition.
  # 
  # Args:
  #   q: query shape == (..., seq_len_q, depth)
  #   k: key shape == (..., seq_len_k, depth)
  #   v: value shape == (..., seq_len_v, depth_v)
  #   mask: Float tensor with shape broadcastable
  #       to (..., seq_len_q, seq_len_k). Defaults to None.
  # 
  # Returns:
  #   output, attention_weights
  # """
  matmul_qk <- tf$matmul(q, k, transpose_b = TRUE)  # (..., seq_len_q, seq_len_k)
  
  
  # scale matmul_qk
  dk <- tfValFl32(tf$shape(k)[-1])
  scaled_attention_logits <- matmul_qk / tf$math$sqrt(dk)
  
  # add the mask to the scaled tensor.
  if(!is.null(mask)){
    scaled_attention_logits <- scaled_attention_logits + (mask * -1e9)
  }
  
  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights <- tf$nn$softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  
  output <- tf$matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  
  return(list(output, attention_weights))
}
#         1.3.6. Multi-head attention                 ####
MultiHeadAttention <- Layer(classname = 'MultiHeadAttention',
                            initialize = function(d_model, num_heads) {
                              super()$`__init__`()
                              self$num_heads  = num_heads
                              self$d_model = d_model
                              self$depth = d_model %/% num_heads
                              self$wq = tf$keras$layers$Dense(d_model)
                              self$wk = tf$keras$layers$Dense(d_model)
                              self$wv = tf$keras$layers$Dense(d_model)
                              self$dense = tf$keras$layers$Dense(d_model)
                              # self$split_heads = function(x, batch_size){
                              #   # """Split the last dimension into (num_heads, depth).
                              #   #    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
                              #   # """
                              #   x = tf$reshape(x, tfVal32(c(batch_size, -1, self$num_heads, self$depth)))
                              # 
                              #   return(tf$transpose(x, perm=tfVal32(c(0, 2, 1, 3))))
                              # }
                            },
                            build = function(input_shape) {
                              # print(class(input_shape))
                              # input_shape1 <- input_shape
                              self$kernel <- self$add_weight(
                                name = "kernel",
                                shape = list(input_shape[[2]], self$d_model),
                                initializer = "uniform",
                                trainable = TRUE
                              )
                            },
                            
                            split_heads = function(x, batch_size){
                              # """Split the last dimension into (num_heads, depth).
                              #    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
                              # """
                              x = tf$reshape(x, tfVal32(c(batch_size, -1, self$num_heads, self$depth)))
                              
                              return(tf$transpose(x, perm=tfVal32(c(0, 2, 1, 3))))
                            },
                            call = function(v, k, q, mask){
                              batch_size <- tf$shape(q)[1]
                              
                              q <- self$wq(q)  # (batch_size, seq_len, d_model)
                              k <- self$wk(k)  # (batch_size, seq_len, d_model)
                              v <- self$wv(v)  # (batch_size, seq_len, d_model)
                              
                              q <- self$split_heads(q, as.numeric(batch_size))  # (batch_size, num_heads, seq_len_q, depth)
                              k <- self$split_heads(k, as.numeric(batch_size))  # (batch_size, num_heads, seq_len_k, depth)
                              v <- self$split_heads(v, as.numeric(batch_size))  # (batch_size, num_heads, seq_len_v, depth)
                              
                              # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
                              # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
                              c(scaled_attention, attention_weights) %<-% scaled_dot_product_attention(q, k, v, mask)
                              
                              scaled_attention <- tf$transpose(scaled_attention, perm=tfVal32(c(0, 2, 1, 3)))  # (batch_size, seq_len_q, num_heads, depth)
                              
                              concat_attention <- tf$reshape(scaled_attention, tfVal32(c(as.numeric(batch_size), -1, self$d_model)))  # (batch_size, seq_len_q, d_model)
                              
                              output <- self$dense(concat_attention)  # (batch_size, seq_len_q, d_model)
                              
                              return(list(output, attention_weights))
                            }
)

#         1.3.7. Point wise feed forward network      ####
point_wise_feed_forward_network <- function(d_model, dff){
  return(tf$keras$Sequential(list(
    tf$keras$layers$Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
    tf$keras$layers$Dense(d_model)  # (batch_size, seq_len, d_model)
  )))
}


#         1.3.8. Encoder layer                        ####
EncoderLayer <- Layer(classname = 'EncoderLayer',
                      initialize = function(d_model, 
                                            num_heads, 
                                            dff, 
                                            rate=0.1) {
                        super()$`__init__`()
                        self$mha = MultiHeadAttention(d_model = d_model, num_heads = num_heads)
                        self$ffn = point_wise_feed_forward_network(d_model, dff)
                        self$layernorm1 = tf$keras$layers$LayerNormalization(epsilon=1e-6)
                        self$layernorm2 = tf$keras$layers$LayerNormalization(epsilon=1e-6)
                        self$dropout1 = tf$keras$layers$Dropout(rate)
                        self$dropout2 = tf$keras$layers$Dropout(rate)
                      },
                      build = function(input_shape) {
                        # print(class(input_shape))
                        # input_shape1 <- input_shape
                        self$kernel <- self$add_weight(
                          name = "kernel",
                          shape = list(input_shape[[2]], self$d_model),
                          initializer = "uniform",
                          trainable = TRUE
                        )
                      },
                      
                      call = function(x, training, mask){
                        attn_output <- self$mha$call(x, k=x, q=x, mask = mask)[[1]] %>%   # (batch_size, input_seq_len, d_model)
                          self$dropout1(training = training)
                        # attn_output <- dropout1(attn_output, training = training)
                        out1 <- self$layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
                        
                        ffn_output <- self$ffn(out1) %>%   # (batch_size, input_seq_len, d_model)
                          self$dropout2(training=training)
                        
                        out2 <- self$layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
                        
                        return(out2)
                        
                      }
)

#         1.3.9. Decoder layer                        ####
DecoderLayer <- Layer(classname = 'DecoderLayer',
                      initialize = function(d_model, 
                                            num_heads, 
                                            dff, 
                                            rate=0.1) {
                        super()$`__init__`()
                        self$mha1 =  MultiHeadAttention(d_model = d_model, num_heads = num_heads)
                        self$mha2 =  MultiHeadAttention(d_model = d_model, num_heads = num_heads)
                        self$ffn = point_wise_feed_forward_network(d_model, dff)
                        self$layernorm1 = tf$keras$layers$LayerNormalization(epsilon=1e-6)
                        self$layernorm2 = tf$keras$layers$LayerNormalization(epsilon=1e-6)
                        self$layernorm3 = tf$keras$layers$LayerNormalization(epsilon=1e-6)
                        self$dropout1 = tf$keras$layers$Dropout(rate)
                        self$dropout2 = tf$keras$layers$Dropout(rate)
                        self$dropout3 = tf$keras$layers$Dropout(rate)
                      },
                      build = function(input_shape) {
                        # print(class(input_shape))
                        # input_shape1 <- input_shape
                        self$kernel <- self$add_weight(
                          name = "kernel",
                          shape = list(input_shape[[2]], self$d_model),
                          initializer = "uniform",
                          trainable = TRUE
                        )
                      },
                      
                      call = function(x, enc_output, training, look_ahead_mask, padding_mask){
                        # enc_output.shape == (batch_size, input_seq_len, d_model)
                        c(attn1, attn_weights_block1) %<-% self$mha1$call(x, k=x, q=x, mask = look_ahead_mask)  # (batch_size, target_seq_len, d_model)
                        attn1 <- self$dropout1(attn1, training=training)
                        
                        out1 <- self$layernorm1(attn1 + x)
                        
                        c(attn2, attn_weights_block2) %<-% self$mha2$call(
                          enc_output, k=enc_output, q=out1, mask = padding_mask)  # (batch_size, target_seq_len, d_model)
                        attn2 <- self$dropout2(attn2, training=training)
                        out2 <- self$layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
                        
                        ffn_output <- self$ffn(out2) %>%   # (batch_size, target_seq_len, d_model)
                          self$dropout3(training=training)
                        # ffn_output <- self$dropout3(ffn_output, training=training)
                        
                        out3 <- self$layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
                        
                        return(list(out3, attn_weights_block1, attn_weights_block2))
                        
                      }
)
#         1.3.10. Encoder                             ####
Encoder <- Layer(classname = 'Encoder',
                 initialize = function(num_layers,
                                       d_model,
                                       num_heads,
                                       dff,
                                       input_vocab_size,
                                       maximum_position_encoding,
                                       rate=0.1) {
                   super()$`__init__`()
                   
                   self$d_model = d_model
                   self$num_layers = num_layers
                   
                   self$embedding = layer_embedding(input_dim = input_vocab_size,
                                                    output_dim = d_model)
                   
                   
                   self$pos_encoding = positional_encoding(maximum_position_encoding, d_model)
                   
                   self$enc_layers = sapply(seq_len(self$num_layers), function(x)
                     EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, rate = rate)
                   )
                   
                   self$dropout = tf$keras$layers$Dropout(rate)
                   
                   
                 },
                 build = function(input_shape) {
                   # print(class(input_shape))
                   # input_shape1 <- input_shape
                   self$kernel <- self$add_weight(
                     name = "kernel",
                     shape = list(input_shape[[2]], self$d_model),
                     initializer = "uniform",
                     trainable = TRUE
                   )
                 },
                 
                 call = function(x, training, mask){
                   seq_length <- as.numeric(tf$shape(x)[2])
                   
                   # adding embedding and position encoding.
                   x <- self$embedding(x) * # (batch_size, input_seq_len, d_model)
                     tf$math$sqrt(tfValFl32(self$d_model)) +
                     self$pos_encoding[, seq_len(seq_length),] %>%
                     self$dropout(training=training)
                   
                   # x <- self$embedding(x) # (batch_size, input_seq_len, d_model)
                   # x <- x * tf$math$sqrt(tfValFl32(self$d_model))
                   # x <- x + self$pos_encoding[, seq_len(seq_length),]
                   # x <- self$dropout(x, training=training)
                   
                   # for(i in seq_len(length(self$enc_layers))){
                   # for(i in 0:(self$num_layers-1)){
                     for(i in seq(self$num_layers)){
                     x <- self$enc_layers[[i]]$call(x, training, mask)
                   }
                   
                   return(x)
                   
                 }
)

#         1.3.11. Decoder                             ####
Decoder <- Layer(classname = 'Decoder',
                 initialize = function(num_layers,
                                       d_model,
                                       num_heads,
                                       dff,
                                       target_vocab_size,
                                       maximum_position_encoding,
                                       rate=0.1) {
                   super()$`__init__`()
                   
                   self$d_model = d_model
                   self$num_layers = num_layers
                   
                   self$embedding = layer_embedding(input_dim = target_vocab_size,
                                                    output_dim = d_model)
                   
                   
                   self$pos_encoding = positional_encoding(maximum_position_encoding, d_model)
                   
                   self$dec_layers = sapply(seq_len(self$num_layers), function(x) 
                     DecoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, rate = rate)
                   )
                   
                   self$dropout = tf$keras$layers$Dropout(rate)
                   
                   
                 },
                 build = function(input_shape) {
                   # print(class(input_shape))
                   # input_shape1 <- input_shape
                   self$kernel <- self$add_weight(
                     name = "kernel",
                     shape = list(input_shape[[2]], self$d_model),
                     initializer = "uniform",
                     trainable = TRUE
                   )
                 },
                 
                 call = function(x, enc_output, training, look_ahead_mask, padding_mask){
                   seq_length <- as.numeric(tf$shape(x)[2])
                   attention_weights <- list()
                   # adding embedding and position encoding.
                   x <- self$embedding(x) * # (batch_size, input_seq_len, d_model)
                     tf$math$sqrt(tfValFl32(self$d_model)) +
                     self$pos_encoding[, seq_len(seq_length),] %>%
                     self$dropout(training=training)
                   
                   # x <- self$embedding(x) # (batch_size, input_seq_len, d_model)
                   # x <- x * tf$math$sqrt(tfValFl32(self$d_model))
                   # x <- x + self$pos_encoding[, seq_len(seq_length),]
                   # x <- self$dropout(x, training=training)
                   
                   
                   # for(i in seq_len(length(self$enc_layers))){
                   # for(i in 0:(self$num_layers-1)){
                   for(i in seq(self$num_layers)){
                     c(x, block1, block2) %<-% self$dec_layers[[i]]$call(x, enc_output, training,
                                                                         look_ahead_mask, padding_mask)
                     
                     
                     attention_weights[[paste0('decoder_layer_', i+1)]] <- list(
                       block1 = block1,
                       block2 = block2
                     )
                   }
                   
                   return(list(x, attention_weights))
                   
                 }
)


#         1.3.12. Transformer                         ####
Transformer <- function(num_layers, 
                        d_model, 
                        num_heads, 
                        dff, 
                        input_vocab_size,
                        target_vocab_size, 
                        pe_input, 
                        pe_target, 
                        rate=0.1,
                        name = 'Transformer'){
  keras_model_custom(name = name, function(self) {
    
    self$encoder = Encoder(num_layers = num_layers, 
                           d_model = d_model, 
                           num_heads = num_heads, 
                           dff = dff, 
                           input_vocab_size = input_vocab_size, 
                           maximum_position_encoding = pe_input, 
                           rate = rate)
    
    self$decoder = Decoder(num_layers = num_layers, 
                           d_model = d_model, 
                           num_heads = num_heads, 
                           dff = dff,
                           target_vocab_size = target_vocab_size, 
                           maximum_position_encoding = pe_target, 
                           rate = rate)
    
    self$final_layer = tf$keras$layers$Dense(target_vocab_size)
    
    self$create_masks = function(inp, tar){
      # Encoder padding mask
      enc_padding_mask <- create_padding_mask(inp)
      
      # Used in the 2nd attention block in the decoder.
      # This padding mask is used to mask the encoder outputs.
      dec_padding_mask <- create_padding_mask(inp)
      
      # Used in the 1st attention block in the decoder.
      # It is used to pad and mask future tokens in the input received by
      # the decoder.
      look_ahead_mask <- create_look_ahead_mask(tf$shape(tar)[2])
      dec_target_padding_mask <- create_padding_mask(tar)
      
      look_ahead_mask = tf$maximum(dec_target_padding_mask, look_ahead_mask)
      
      return(list(enc_padding_mask, look_ahead_mask, dec_padding_mask))
    }
    
    
    
    self$main <- function(inputs, training){
      # Keras models prefer if you pass all your inputs in the first argument
      c(inp, tar) %<-% inputs
      
      c(enc_padding_mask, look_ahead_mask, dec_padding_mask) %<-% self$create_masks(inp, tar)
      
      enc_output <- self$encoder$call(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
      
      # dec_output.shape == (batch_size, tar_seq_len, d_model)
      c(dec_output, attention_weights) %<-% self$decoder$call(
        tar, 
        enc_output, 
        training, 
        look_ahead_mask, 
        dec_padding_mask)
      
      final_output <- self$final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
      
      return(list(final_output, attention_weights))
      
      
    }
  })
}

transformer = Transformer(
  num_layers=num_layers,
  d_model=d_model,
  num_heads=num_heads,
  dff=dff,
  input_vocab_size=src_vocab_size,
  target_vocab_size=target_vocab_size,
  pe_input=pe_input,
  pe_target=pe_target,
  rate=dropout_rate)


#         1.3.13. Losses and indicators               ####
# they are appearing here already in the file because we want to watch how
# the network learns

#                1.3.13.1. Data for graph             ####
graph_data <- na.omit(data.table(epoch      = as.numeric(NA),
                                 train_perp = as.numeric(NA),
                                 val_perp   = as.numeric(NA)))

theme_dt <- function (color = "black") {
  
  if(color == "black") color2 <- "white" else color2 <- "black"
  
  theme(      text              = element_text (family = "Panton",
                                                color  = color2   ),
              rect              = element_rect (fill   = color    ),
              line              = element_line (color  = color    ),
              title             = element_text (
                face   = "bold"   ),
              
              legend.position   = "bottom",
              legend.title      = element_text (
                color = color2,
                face  = "bold"    ),
              legend.key        = element_rect (fill  = color     ),
              legend.background = element_rect (fill  = color     ),
              legend.text       = element_text (
                color = color2,
                face  = "bold"    ),
              
              panel.background  = element_rect (fill = color),
              panel.grid        = element_blank(),
              
              
              axis.text         = element_text (
                color = color2    ),
              plot.caption      = element_text (
                face  = "italic",
                color = color2)
              
  )
}

graph <- function(data){
  print(    ggplotly(
    ggplot(data, aes(x = epoch)) + 
      geom_line(aes(y = train_perp, colour = 'train_perp')) + 
      geom_point(aes(y = train_perp, colour = 'train_perp'), size = 2.5) + 
      geom_line(aes(y = val_perp, colour = 'val_perp')) + 
      geom_point(aes(y = val_perp, colour = 'val_perp'), size = 2.5) + 
      theme_dt() +
      theme (
        text = element_text(family = "Panton"),
        panel.grid.major.x = element_line(colour = "white", linetype = "dotted"),
        panel.grid.major.y = element_line(colour = "white", linetype = "dotted"),
        legend.position = "bottom",
        legend.box.just = "bottom",
        legend.direction = "horizontal",
      ) +
      labs(title = 'Perplexity',
           x = 'Epoch',
           y = 'Perplexity value',
           caption  = "Bogdan Tewnalany") + 
      scale_color_manual (name = "Type:",
                          values = c("train_perp" = '#6fb3a7',"val_perp" = '#e69c18'),
                          labels = c("train_perp" = "Train perpl","val_perp" = "Valid perpl")) 
    
  )
  )
}

plot_attention <-
  function(attention_matrix,
           words_sentence,
           words_result) {
    # melted <- melt(attention_matrix)
    ggplot(data = attention_matrix, aes(
      x = factor(Var2),
      y = factor(Var1),
      fill = value
    )) +
      geom_tile() + 
      facet_wrap(~ Var3, ncol = 3) + 
      scale_fill_viridis() + guides(fill = "none") +
      theme(axis.ticks = element_blank()) +
      xlab("") +
      ylab("") +
      scale_x_discrete(labels = words_sentence, position = "top") +
      scale_y_discrete(labels = words_result) +
      theme(aspect.ratio = 1)
  }

#                1.3.13.2. Evaluation and translation ####
# translator <-
#   function(sentence) {
#     sentence <- preprocess_sentence(sentence)
#     input <- sentence2digits(sentence, index_df = src_index)
#     input <-
#       pad_sequences(list(input), maxlen = src_maxlen,  padding = "post") %>% 
#       k_constant()
#     
#     result <- ""
#     result_list <- list(word2index("<start>", target_index))
# 
#     dec_input <- k_expand_dims(result_list)
#     
#     # dec_input <- tf$TensorArray(dtype=tf$int64, size=tfVal64(0), dynamic_size=TRUE)
#     # dec_input <- dec_input$write(tfVal64(0), tfVal64(word2index("<start>", target_index)))
# 
#     for (t in seq_len(target_maxlen)) {
#       # output <- dec_input$stack() %>% k_expand_dims() %>% tf$transpose()
#       preds <- transformer$main(list(input, output), training=FALSE)[[1]]
#       # c(preds, attention_weights) %<-% transformer$main(list(input, dec_input), training=FALSE)
#       
#       predictions <- preds[, -1, ]
# 
#       # Аз бир бирде болуўчу
#       pred_idx <- tf$random$categorical(k_exp(predictions), num_samples = 1L)[1, 1] %>% as.double()
# 
#       # Тюз
#        # pred_idx <- tf$argmax(predictions, axis=tfVal64(-1)) %>% as.double()
#       
#       
#       pred_word <- index2word(pred_idx, target_index)
#       
#       result_list[t+1] <- pred_idx
#       
#       if (pred_word == '<stop>') {
#         break
#       } else {
#         # dec_input <- k_expand_dims(list(pred_idx))
#         dec_input <- k_expand_dims(result_list) %>% tf$transpose()
#         # dec_input <- dec_input$write(tfVal64(t), tfVal64(pred_idx))
#       }
#     }
#       
#       tokens <- pad_sequences(list(result_list), maxlen = target_maxlen,  padding = "post") %>% 
#         k_constant()
#     
#       result <- sapply(result_list, function(index) index2word(index, index_df = target_index)) %>% paste0(collapse = ' ')
#     
#     attention_weights <- transformer$main(list(input, tokens), training=FALSE)[[2]]
#     
#     
#     return(list(str_trim(result), sentence, attention_weights, tokens))
#   }

toModel <- function(string){
  string %>% 
    str_replace_all('\\bсебеп|\\bсебеб', 'себеb') %>% 
    str_replace_all('\\bСебеп|\\bСебеб', 'Себеb') %>% 
    str_replace_all('\\bтюйюл', 'тюл') %>% 
    str_replace_all('\\bТюйюл', 'Тюл') %>% 
    str_replace_all('\\bуку', 'гылын qуш') %>% 
    str_replace_all('\\bУку', 'Гылын qуш') %>% 
    str_replace_all('\\bхораз', 'гугурукку') %>%
    str_replace_all('\\bХораз', 'Гугурукку') %>%
    str_replace_all('\\bюзмез', 'qум') %>% 
    str_replace_all('\\bЮзмез', 'Qум') %>% 
    str_replace_all('\\bарап|\\bараб', 'араb') %>% 
    str_replace_all('\\bАрап|\\bАраб', 'Араb') %>% 
    str_replace_all('\\bжиля|\\bжыла|\\bджыла', 'jыла') %>% 
    str_replace_all('\\bЖиля|\\bЖыла|\\bДжыла', 'Jыла') %>% 
    str_replace_all('\\bярабий|\\bарабий', 'арабин') %>% 
    str_replace_all('\\bЯрабий|\\bАрабий', 'Арабин') %>% 
    str_replace_all('нтта', 'нтда') %>% 
    str_replace_all('ртте', 'ртде') %>% 
    str_replace_all('\\bжамауат|\\bжамагъат|\\bджамагъат|\\bжамауат', 'jамаgат') %>%
    str_replace_all('\\bЖамауат|\\bЖамагъат|\\bДжамагъат|\\bЖамаўат', 'Jамаgат') %>%
    str_replace_all('\\bшуёх', 'шох') %>% 
    str_replace_all('\\bШуёх', 'Шох') %>% 
    str_replace_all('\\bшёндю|\\bбусагъат', 'бусаgат') %>% 
    str_replace_all('\\bШёндю|\\bБусагъат', 'Бусаgат') %>% 
    str_replace_all('\\bугъай|\\bогъай', 'оgай') %>% 
    str_replace_all('\\bУгъай|\\bОгъай', 'Оgай') %>% 
    # str_replace_all('\\bтерк', 'тез') %>% 
    # str_replace_all('\\bтерк', 'тез') %>% 
    str_replace_all('\\bсанга|\\bсеннге|\\bсенге', 'сенnе') %>% 
    str_replace_all('\\bСанга|\\bСеннге|\\bСенге', 'Сенnе') %>% 
    str_replace_all('\\bманга|\\bменнге|\\bменге', 'менnе') %>% 
    str_replace_all('\\bМанга|\\bМеннге|\\bМенге', 'Менnе') %>% 
    str_replace_all('\\bаякъ жол|\\bаякъ джол|\\bджахтана|\\bжахтана', 'jахтана') %>% 
    str_replace_all('\\bАякъ жол|\\bАякъ джол|\\bДжахтана|\\bЖахтана', 'Jахтана') %>% 
    str_replace_all('\\bкъамж|\\bкъамыж', 'qамыzh') %>% 
    str_replace_all('\\bКъамж|\\bКъамыж', 'Qамыzh') %>% 
    str_replace_all('\\bкъымыж|\\bкъымыж', 'qымыzh') %>% 
    str_replace_all('\\bКъымыж|\\bКъымыж', 'Qымыzh') %>% 
    str_replace_all('\\bхау\\b|\\bхаў\\b', 'хо') %>% 
    str_replace_all('\\bХау\\b|\\bХаў\\b', 'Хо') %>% 
    str_replace_all('\\bуа\\b|\\bўа\\b', 'wa') %>% 
    str_replace_all('\\bУа\\b|\\bЎа\\b', 'Wa') %>% 
    str_replace_all('п\\b|б\\b', 'b') %>% 
    str_replace_all('къ', 'q') %>% 
    str_replace_all('Къ|КЪ', 'Q') %>% 
    str_replace_all('гъ', 'g') %>% 
    str_replace_all('Гъ|ГЪ', 'G') %>% 
    str_replace_all('ц', 'ч') %>% 
    str_replace_all('Ц', 'Ч') %>% 
    str_replace_all('дж', 'j') %>% 
    str_replace_all('Дж|ДЖ', 'J') %>% 
    str_replace_all('ж', 'j') %>% 
    str_replace_all('Ж', 'J') %>% 
    str_replace_all('ф', 'п') %>% 
    str_replace_all('Ф', 'П') %>% 
    str_replace_all('(?<=[аыоуэеиёюя])у(?=[аыоуэеиёюя])|(?<=[аыоуэеиёюя])ў(?=[аыоуэеиёюя])|(?<=[АЫОУЭЕИЁЮЯ])у(?=[АЫОУЭЕИЁЮЯ])|(?<=[АЫОУЭЕИЁЮЯ])ў(?=[АЫОУЭЕИЁЮЯ])', 'w') %>% 
    str_replace_all('(?<=[аыоуэеиёюя])у|(?<=[аыоуэеиёюя])ў|(?<=[АЫОУЭЕИЁЮЯ])у|(?<=[АЫОУЭЕИЁЮЯ])ў', 'w') %>% 
    # str_replace_all('у(?=[аыоуэеиёюя])|ў(?=[аыоуэеиёюя])|у(?=[АЫОУЭЕИЁЮЯ])|ў(?=[АЫОУЭЕИЁЮЯ])', 'w') %>% 
    # str_replace_all('У(?=[аыоуэеиёюя])|Ў(?=[аыоуэеиёюя])|У(?=[АЫОУЭЕИЁЮЯ])|Ў(?=[АЫОУЭЕИЁЮЯ])', 'W') %>% 
    str_replace_all('zh', 'ж') %>% 
    str_replace_all('нг', 'n') %>% 
    str_replace_all('Нг|НГ', '  N')
  # str_replace_all('\\bчап', 'чаб')  # чаб бла
}
fromModel <- function(string, 
                      dialect = 'qrc' # 'hlm', 'mqr'
){
  if(dialect == 'qrc'){
    string %>% 
      str_replace_all('\\bтюйюл', 'тюл') %>% 
      str_replace_all('\\bТюйюл', 'Тюл') %>% 
      str_replace_all('\\bуку', 'гылын  qуш') %>% 
      str_replace_all('\\bУку', 'Гылын  qуш') %>% 
      str_replace_all('\\bхораз', 'гугурукку') %>%
      str_replace_all('\\bХораз', 'Гугурукку') %>%
      str_replace_all('\\bюзмез', 'qум') %>% 
      str_replace_all('\\bЮзмез', 'Qум') %>% 
      str_replace_all('\\bjиля', 'jыла') %>% 
      str_replace_all('\\bJиля', 'Jыла') %>% 
      str_replace_all('\\bярабий|\\bарабий', 'арабин') %>% 
      str_replace_all('\\bЯрабий|\\bАрабий', 'Арабин') %>% 
      str_replace_all('нтта', 'нтда') %>% 
      str_replace_all('ртте', 'ртде') %>% 
      str_replace_all('\\bjамауат|\\bjамаwат', 'jамаgат') %>%
      str_replace_all('\\bJамауат|\\bJамаwат', 'Jамаgат') %>%
      str_replace_all('\\bшуёх', 'шох') %>% 
      str_replace_all('\\bШуёх', 'Шох') %>% 
      str_replace_all('\\bшёндю', 'бусаgат') %>% 
      str_replace_all('\\bШёндю', 'Бусаgат') %>% 
      str_replace_all('\\bуgай', 'оgай') %>% 
      str_replace_all('\\bУgай', 'Оgай') %>% 
      # str_replace_all('\\bтерк', 'тез') %>% 
      str_replace_all('\\bсаnа|\\bсеnе', 'сенnе') %>% 
      str_replace_all('\\bСаnа|\\bСеnе', 'Сенnе') %>% 
      str_replace_all('\\bмаnа|\\bмеnе', 'менnе') %>% 
      str_replace_all('\\bМаnа|\\bМеnе', 'Менnе') %>% 
      str_replace_all('\\bаяq jол', 'jахтана') %>% 
      str_replace_all('\\bАяq jол', 'Jахтана') %>% 
      str_replace_all('b', 'б') %>% 
      str_replace_all('q', 'къ') %>% 
      str_replace_all('Q', 'Къ') %>% 
      str_replace_all('g', 'гъ') %>% 
      str_replace_all('G', 'Гъ') %>% 
      str_replace_all('j', 'дж') %>% 
      str_replace_all('J', 'Дж') %>% 
      str_replace_all('w', 'ў') %>% 
      str_replace_all('W', 'Ў') %>% 
      str_replace_all('n', 'нг') %>% 
      str_replace_all('N', 'Нг')
  } else if(dialect == 'hlm'){
    string %>% 
      str_replace_all('\\bтюл', 'тюйюл') %>% 
      str_replace_all('\\bТюл', 'Тюйюл') %>% 
      str_replace_all('\\bгылын  qуш', 'уку') %>% 
      str_replace_all('\\bГылын  qуш', 'Уку') %>% 
      str_replace_all('\\bгугурукку', 'хораз') %>%
      str_replace_all('\\bГугурукку', 'Хораз') %>%
      str_replace_all('\\bqум', 'юзмез') %>% 
      str_replace_all('\\bQум', 'Юзмез') %>% 
      str_replace_all('\\bjыла', 'jиля') %>% 
      str_replace_all('\\bJыла', 'Jиля') %>% 
      str_replace_all('\\bарабин|\\bарабий', 'ярабий') %>% 
      str_replace_all('\\bАрабин|\\bАрабий', 'Ярабий') %>% 
      str_replace_all('нтда', 'нтта') %>% 
      str_replace_all('ртде', 'ртте') %>% 
      str_replace_all('\\bjамаgат', 'jамаwат') %>%
      str_replace_all('\\bJамаgат', 'Jамаwат') %>%
      str_replace_all('\\bшох', 'шуёх') %>% 
      str_replace_all('\\bШох', 'Шуёх') %>% 
      str_replace_all('\\bбусаgат', 'шёндю') %>% 
      str_replace_all('\\bБусаgат', 'Шёндю') %>% 
      str_replace_all('\\bоgай', 'уgай') %>% 
      str_replace_all('\\bОgай', 'Уgай') %>% 
      str_replace_all('\\bтез', 'терк') %>% 
      str_replace_all('\\bсенnе|\\bсеnе', 'саnа') %>% 
      str_replace_all('\\bСенnе|\\bСеnе', 'Саnа') %>% 
      str_replace_all('\\bменnе|\\bмеnе', 'маnа') %>% 
      str_replace_all('\\bМенnе|\\bМеnе', 'Маnа') %>% 
      str_replace_all('\\bjахтана', 'аяq jол') %>% 
      str_replace_all('\\bJахтана', 'аяq jол') %>% 
      str_replace_all('\\bхо\\b', 'хаw') %>% 
      str_replace_all('\\bХо\\b', 'Хаw') %>% 
      str_replace_all('b', 'п') %>% 
      str_replace_all('q', 'къ') %>% 
      str_replace_all('Q', 'Къ') %>% 
      str_replace_all('g', 'гъ') %>% 
      str_replace_all('G', 'Гъ') %>% 
      str_replace_all('j', 'ж') %>% 
      str_replace_all('J', 'Ж') %>% 
      str_replace_all('w', 'ў') %>% 
      str_replace_all('W', 'Ў') %>% 
      str_replace_all('n', 'нг') %>% 
      str_replace_all('N', 'Нг')
    
  } else if(dialect == 'mqr'){
    string %>% 
      str_replace_all('\\bтюл', 'тюйюл') %>% 
      str_replace_all('\\bТюл', 'Тюйюл') %>% 
      str_replace_all('\\bгылын  qуш', 'уку') %>% 
      str_replace_all('\\bГылын  qуш', 'Уку') %>% 
      str_replace_all('\\bгугурукку', 'хораз') %>%
      str_replace_all('\\bГугурукку', 'Хораз') %>%
      str_replace_all('\\bqум', 'юзмез') %>% 
      str_replace_all('\\bQум', 'Юзмез') %>% 
      str_replace_all('\\bjыла', 'jиля') %>% 
      str_replace_all('\\bJыла', 'Jиля') %>% 
      str_replace_all('\\bарабин|\\bарабий', 'ярабий') %>% 
      str_replace_all('\\bАрабин|\\bАрабий', 'Ярабий') %>% 
      str_replace_all('нтда', 'нтта') %>% 
      str_replace_all('ртде', 'ртте') %>% 
      str_replace_all('\\bjамаgат', 'jамаwат') %>%
      str_replace_all('\\bJамаgат', 'Jамаwат') %>%
      str_replace_all('\\bшох', 'шуёх') %>% 
      str_replace_all('\\bШох', 'Шуёх') %>% 
      str_replace_all('\\bбусаgат', 'шёндю') %>% 
      str_replace_all('\\bБусаgат', 'Шёндю') %>% 
      str_replace_all('\\bоgай', 'уgай') %>% 
      str_replace_all('\\bОgай', 'Уgай') %>% 
      str_replace_all('\\bтез', 'терк') %>% 
      str_replace_all('\\bсенnе|\\bсеnе', 'саnа') %>% 
      str_replace_all('\\bСенnе|\\bСеnе', 'Саnа') %>% 
      str_replace_all('\\bменnе|\\bмеnе', 'маnа') %>% 
      str_replace_all('\\bМенnе|\\bМеnе', 'Маnа') %>% 
      str_replace_all('\\bjахтана', 'аяq jол') %>% 
      str_replace_all('\\bJахтана', 'аяq jол') %>% 
      str_replace_all('\\bхо\\b', 'хаw') %>% 
      str_replace_all('\\bХо\\b', 'Хаw') %>% 
      str_replace_all('b', 'п') %>% 
      str_replace_all('q', 'къ') %>% 
      str_replace_all('Q', 'Къ') %>% 
      str_replace_all('g', 'гъ') %>% 
      str_replace_all('G', 'Гъ') %>% 
      str_replace_all('j', 'з') %>% 
      str_replace_all('J', 'З') %>% 
      str_replace_all('w', 'ў') %>% 
      str_replace_all('W', 'Ў') %>% 
      str_replace_all('n', 'нг') %>% 
      str_replace_all('N', 'Нг') %>% 
      str_replace_all('ч', 'ц') %>% 
      str_replace_all('Ч', 'Ц') %>% 
      str_replace_all('п', 'ф') %>% 
      str_replace_all('П', 'Ф') %>% 
      str_replace_all('къ\\b|гъ\\b', 'х')
  }
  
}

translator <-
  function(sentence, need_attention = FALSE) {
    # sentence <- preprocess_sentence(sentence)
    # input <- sentence2digits(sentence, index_df = src_index)
    # input <-
    #   pad_sequences(list(input), maxlen = src_maxlen,  padding = "post") %>% 
    #   k_constant()
    
    input <- bpe_encode(model = model_src_bpe, 
                        x     = sentence, 
                        type  = "ids", 
                        bos   = TRUE, 
                        eos   = TRUE) %>% 
      pad_sequences(maxlen = src_maxlen,  padding = "post") %>% 
      k_constant()
    
    
    # Encoder and decoder padding mask 
    enc_padding_mask <- create_padding_mask(input)
    # Encoder part
    enc_output <- transformer$encoder$call(input, training = FALSE, enc_padding_mask) 
    
    result <- ""
    
    result_list <- list(
      as.data.table(model_trg_bpe$vocabulary)['<BOS>', on = 'subword']$id
    )
    
    end_id <- as.data.table(model_trg_bpe$vocabulary)['<EOS>', on = 'subword']$id
    
    dec_input <- k_expand_dims(result_list)
    
    # dec_input <- tf$TensorArray(dtype=tf$int64, size=tfVal64(0), dynamic_size=TRUE)
    # dec_input <- dec_input$write(tfVal64(0), tfVal64(word2index("<start>", target_index)))
    
    for (t in seq_len(trg_maxlen)) {
    # t <- 1
    # while(TRUE){
      # output <- dec_input$stack() %>% k_expand_dims() %>% tf$transpose()
      # preds <- transformer$main(list(input, output), training=FALSE)[[1]]
      
      # dec_output.shape == (batch_size, tar_seq_len, d_model)
      look_ahead_mask <- create_look_ahead_mask(tf$shape(dec_input)[2])
      dec_target_padding_mask <- create_padding_mask(dec_input)
      
      look_ahead_mask <- tf$maximum(dec_target_padding_mask, look_ahead_mask)
      
      dec_output <- transformer$decoder$call(
        dec_input, 
        enc_output, 
        training = FALSE, 
        look_ahead_mask, 
        padding_mask = enc_padding_mask)[[1]]
      
      preds <- transformer$final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
      
      
      predictions <- preds[, -1, ]
      
      # Аз бир бирде болуўчу
      pred_idx <- tf$random$categorical(k_exp(predictions), num_samples = 1L)[1, 1] %>% as.double()
      
      # Тюз
      # pred_idx <- tf$argmax(predictions, axis=tfVal64(-1)) %>% as.double()
      
      
      # pred_word <- index2word(pred_idx, target_index)
      
      result_list[t+1] <- pred_idx
      t <- t + 1
      if (pred_idx == end_id) {
        break
      } else {
        # dec_input <- k_expand_dims(list(pred_idx))
        dec_input <- k_expand_dims(result_list) %>% tf$transpose()
        # dec_input <- dec_input$write(tfVal64(t), tfVal64(pred_idx))
      }
    }
    
    # result <- sapply(result_list, function(index) index2word(index, index_df = target_index)) %>% paste0(collapse = ' ') %>%  str_replace_all(pattern = '<start> | <stop>', replacement = '')
    
    
    
    if(need_attention){
      
      result <- 
        bpe_decode(model = model_trg_bpe,
                   x = as.integer(unlist(result_list))) %>% 
        str_replace_all(pattern = '<BOS>|<EOS>|<PAD>', replacement = '') %>% 
        str_squish() %>% 
        fromModel()
        # str_replace_all("\\ ([:punct:])", "\\1")
      
      
      tokens <- pad_sequences(list(result_list), maxlen = trg_maxlen,  padding = "post") %>% 
        k_constant()
      
      
      look_ahead_mask <- create_look_ahead_mask(tf$shape(tokens)[2])
      dec_target_padding_mask <- create_padding_mask(tokens)
      
      look_ahead_mask <- tf$maximum(dec_target_padding_mask, look_ahead_mask)
      
      attention_weights <- transformer$decoder$call(
        tokens, 
        enc_output, 
        training = FALSE, 
        look_ahead_mask, 
        padding_mask = enc_padding_mask)[[2]]
      
      # attention_weights <- transformer$main(list(input, tokens), training=FALSE)[[2]]
      
      # sentence <- str_replace_all(string = sentence, pattern = '<start> | <stop>', replacement = '')
      
      return(list(str_trim(result), sentence, attention_weights, tokens))
      
    } else {
      result <- 
        bpe_decode(model = model_trg_bpe,
                   x = as.integer(unlist(result_list))) %>% 
        str_replace_all(pattern = '<BOS>|<EOS>|<PAD>', replacement = '') %>% 
        str_squish() %>% 
        fromModel()
        # str_replace_all("\\ ([:punct:])", "\\1")
      
      return(result)
    }
  }

# sentence <- validation_sample[[1]][1]
# result <- validation_sample[[1]][2]
evaluate_validation <- function() {
  iter <- make_iterator_one_shot(validation_dataset)
  batch <- iterator_get_next(iter)
  loss <- 0
  x <- batch[[1]]
  # y <- tf$cast(batch[[2]],  dtype=tf$int64)
  y <- batch[[2]]
  
  preds <- transformer$main(list(x, y), training = TRUE)[[1]]
  loss <- loss_function(y, preds)
  
  return(loss / k_cast_to_floatx(dim(y)[2]))
}


introduction <- function(sentence) {
  # BLEU - compares only words and phrases
  # ChrF++ – also compares pieces of several letters (this helps if the word is misspelled or in a different case)
  # NIST - correctly matching a rare n-gram improves your score more than a correct match to a regular n-gram.
  
  if(is.list(sentence) | (length(sentence) > 1)){
    
    sentence <- unlist(sentence)
    c(result, input_sentence, attention_weights, tokens) %<-% translator(sentence[[SRC_LANG]], need_attention = TRUE)
    
    paste0("Input:                 ", sentence[[SRC_LANG]], '\n',
           "Predicted translation: ", result, '\n',
           "Right translation:     ", sentence[[TRG_LANG]], '\n',
           "BLUE:                  ", try(nltk$translate$bleu(references = sentence[[TRG_LANG]], hypothesis = result) %>% round(7)), '\n',
           "NIST:                  ", try(nltk$translate$nist(references = sentence[[TRG_LANG]], hypothesis = result) %>% round(7)), '\n',
           "ChrF++:                ", try(nltk$translate$chrf(reference = sentence[[TRG_LANG]], hypothesis = result) %>% round(7)), '\n\n'
    ) %>% cat()
  } else {
    c(result, input_sentence, attention_weights, tokens) %<-% translator(sentence, need_attention = TRUE)
    
    paste0("Input:                 ", sentence, '\n',
           "Predicted translation: ", result, '\n'
           # "Right translation:     ", sentence[2], '\n',
           # "BLUE:                  ", nltk$translate$bleu(references = sentence[2], hypothesis = result) %>% round(7), '\n',
           # "NIST:                  ", nltk$translate$nist(references = sentence[2], hypothesis = result) %>% round(7), '\n',
           # "ChrF++:                ", nltk$translate$chrf(reference = sentence[2], hypothesis = result) %>% round(7), '\n\n'
    ) %>% cat()
  }
  # print(paste0("Input: ",  sentence))
  # print(paste0("Predicted translation: ", result))
  
  
  attention_matrix <-
    map_dfr(seq_len(num_heads), function(x){
      suppressWarnings(melt(attention_weights$decoder_layer_4$block2[1,x,,] %>% as.matrix())) %>% as.data.table() %>% 
        .[, Var3 := x]
    }
    )
  
  try(plot_attention(attention_matrix, 
                     words_sentence = str_split(sentence, " ")[[1]], 
                     words_result = str_split(result, " ")[[1]]))
}



#    1.4. Training                                    ####
#         1.4.1. Training parametres                  ####
n_epochs <- 15
n_non_improve_val_perplex <- 12

# encoder_init_hidden <- map(seq_len(2), function(x) k_zeros(c(batch_size, gru_units), dtype = NULL))

#         1.4.2. Training loop                        ####

# tf$config$list_physical_devices(device_type='GPU')



for (epoch in seq_len(n_epochs)) {
  total_loss <- 0
  total_loss_val <- 0
  iteration <- 0
  # saved best result
  last_val_best_loss_perpl <- Inf
  perpl_val <- 0
  improve_iter <- 0
  
  
  iter <- make_iterator_one_shot(train_dataset)
  
  try(until_out_of_range({
    batch <- iterator_get_next(iter)
    loss <- 0
    x <- batch[[1]]
    # y <- tf$cast(batch[[2]],  dtype=tf$int64)
    y <- batch[[2]]
    iteration <- iteration + 1
    
    y_inp = y[, 1:(dim(y)[2]-1)]
    y_real = y[, 2:dim(y)[2]]
    
    with(tf$GradientTape() %as% tape, {
      
      preds <- transformer$main(list(x, y_inp), training = TRUE)[[1]]
      loss <- loss_function(y_real, preds)
    })
    total_loss <-
      total_loss + loss / k_cast_to_floatx(dim(y)[2])
    
    total_loss_val <- total_loss_val + try(evaluate_validation())
    
    paste0(
      "Batch loss (epoch/batch/all batches): ",
      epoch,
      "/",
      iteration,
      "/",
      ceiling(nrow(x_train)/batch_size),
      ": ",
      (loss / k_cast_to_floatx(dim(y)[2])) %>% as.double() %>% round(4),
      "; ",
      "Perplexity: ",
      exp((loss / k_cast_to_floatx(dim(y)[2]))) %>% as.double() %>% round(4),
      "\n"
    ) %>% cat()
    
    gradients <- tape$gradient(loss, transformer$trainable_variables)
    
    optimizer$apply_gradients(purrr::transpose(list(gradients, transformer$trainable_variables)))
    
  }))
  
  
  paste0(
    "Total loss (epoch): ",
    epoch,
    ": ",
    (total_loss / k_cast_to_floatx(buffer_size)) %>% as.double() %>% round(4),
    "; ",
    "Perplexity: ",
    exp((total_loss / k_cast_to_floatx(buffer_size))) %>% as.double() %>% round(4),
    "; ",
    "Val loss: ",
    (total_loss_val / k_cast_to_floatx(buffer_size)) %>% as.double() %>% round(4),
    "; ",
    "Val perplexity: ",
    exp((total_loss_val / k_cast_to_floatx(buffer_size))) %>% as.double() %>% round(4),
    "\n"
  ) %>% cat()
  
  perpl_val <- exp((total_loss_val / k_cast_to_floatx(buffer_size))) %>% as.double() %>% round(4)
  
  # Save best result
  if(perpl_val < last_val_best_loss_perpl){
    
    transformer$save_weights(filepath = TRANSFORMER_FILE)
    
    last_val_loss_perpl <- perpl_val
    improve_iter <- epoch
    
  }
  
  graph_data <- rbindlist(list(graph_data, data.table(epoch      = epoch,
                                                      train_perp = exp((total_loss / k_cast_to_floatx(buffer_size))) %>% as.double() %>% round(4),
                                                      val_perp   = exp((total_loss_val / k_cast_to_floatx(buffer_size))) %>% as.double() %>% round(4))))
  
  # Graph 
  print(graph(graph_data))
  
  # Case
  # cat('Train:\n')
  # walk(train_sentences[1:5], function(pair)
  #   introduction(pair[1]))
  # cat('Validation:\n')
  # walk(validation_sample, function(pair)
  #   introduction(pair[1]))
  
  # cat('Train:\n')
  # walk(train_sentences[1:5], 
  #      function(pair) try(introduction(pair)))
  # cat('Validation:\n')
  # walk(validation_sample, function(pair)
  #   try(introduction(pair)))
  
  cat('Test:\n')
  walk(test_pairs[1:5][[SRC_LANG]], 
       function(pair) try(introduction(pair)))

  
  # if improvement there is no more, than <n_non_improve_val_perplex>, break train process
  if(epoch - improve_iter > n_non_improve_val_perplex){ 
    cat('\n Break training in epoch: ', epoch)
    break
  }
}

# encoder$load_weights(filepath = file.path(this.dir(), 'encoder_ing-germ/encoder_ing-germ'))
# decoder$load_weights(filepath = file.path(this.dir(), 'decoder_ing-germ/decoder_ing-germ'))


# parametres_count_encoder <- sum(sapply(encoder$layers, function(x) x$count_params())) + sum(sapply(decoder$layers, function(x) x$count_params()))

number_params <- sapply(transformer$weights, function(x)  prod(dim(x)) + last(dim(x))) %>% sum()


# plot a mask
example_sentence <- 'Your dogs are very awful animals'#  train_sentences[[1]]
translator(example_sentence)

# save_model_weights_tf(object = encoder$weights, filepath = file.path(this.dir(), 'decoder_ing-germ'))
# save_model_weights_tf(encoder, filepath = file.path(this.dir(), 'encoder_ing-germ'))

# encoder$save_weights(filepath = file.path(this.dir(), 'encoder_ing-germ/encoder_ing-germ'))
# decoder$save_weights(filepath = file.path(this.dir(), 'decoder_ing-germ/decoder_ing-germ'))
# 
# encoder$load_weights(filepath = file.path(this.dir(), 'encoder_ing-germ/encoder_ing-germ'))
# decoder$load_weights(filepath = file.path(this.dir(), 'decoder_ing-germ/decoder_ing-germ'))

transformer$save_weights(filepath = file.path(this.dir(), 'transformer_ing-dutch/transformer_ing-dutch'))




transformer$load_weights(filepath = file.path(this.dir(), 'transformer_ing-dutch/transformer_ing-dutch'))


# 2. Translating                                      ####
#    2.0. Libraries                                   ####
library(tensorflow)
library(keras)
library(tfdatasets)
library(this.path)
library(purrr)
library(stringr)
library(reshape2)
library(viridis)
library(ggplot2)
library(tibble)
library(plotly)
library(data.table)
library(reticulate)
library(tokenizers.bpe)
# library(R6)
# reticulate::py_install('nltk', pip = TRUE)
nltk = reticulate::import('nltk')

# dir_path <- paste0(this.dir(), '/')

#    2.1. Hyperparameters / variables                 ####
#         2.1.1. BPE                                  ####
src    <- 'english'
target <- 'dutch'

MODEL_PATH_SRC    <- file.path(this.dir(), 'BPE', paste0("youtokentome_src_", src, ".bpe"))
MODEL_PATH_TARGET <- file.path(this.dir(), 'BPE', paste0("youtokentome_target_", target, ".bpe"))


model_src <- bpe_load_model(MODEL_PATH_SRC)

model_target <- bpe_load_model(MODEL_PATH_TARGET)

#         2.1.2. Hyperparameters / variables          ####
batch_size <- 32
# embedding_dim <- 64
# gru_units <- 256
# enc_dropout = 0.1
# dec_dropout = 0.1
src_maxlen    <- 13
target_maxlen <- 22


src_vocab_size    <- model_src$vocab_size
target_vocab_size <- model_target$vocab_size

num_layers <- 6 # Количество энкодер и декодер слоёв
d_model <- 256 # Количество нейронов в слое (или глубина внимания) (Эм иги 512, компьютерим тарталмайды)
dff <- 1024 # Расстояние позиций (токенов) для слоя позиционно-зависимого прямого распространения (Эм иги 2048, компьютерим тарталмайды)
num_heads <- 8 # Кол-во голов внимания
dropout_rate <- 0.1 # Дропаут
pe_input <- dff
pe_target <- dff

tfVal32 <- function(x){
  tf$cast(x, dtype=tf$int32)
}
tfVal64 <- function(x){
  tf$cast(x, dtype=tf$int64)
}
tfValFl32 <- function(x){
  tf$cast(x, dtype=tf$float32)
}
tfValFl64 <- function(x){
  tf$cast(x, dtype=tf$float64)
}

#    2.2. Model                                       ####

#         2.2.1. Positional encoding                  ####
# Attachments represent the token in a d-dimensional space, where tokens with a similar value will be closer to each other. But attachments do not encode the relative position of the tokens in the sentence. Thus, after adding the positional encoding, the tokens will be closer to each other based on the similarity of their meaning and their position in the sentence, in d-dimensional space.

get_angles <- function(pos, i, d_model){
  angle_rates = 1 / (10000 ** ((2 * (i %/% 2)) / as.double(d_model)))
  return(pos %*% angle_rates)
}

positional_encoding <- function(position, d_model){
  angle_rads <- get_angles(pos = (seq(position)-1) %>% as.matrix(),
                           i = (seq(d_model)-1) %>% as.matrix() %>% t(),
                           d_model = d_model)
  # apply sin to even indices in the array; 2i
  angle_rads[, seq(1, dim(angle_rads)[2], by = 2)] <- sin(angle_rads[, seq(1, dim(angle_rads)[2], by = 2)])
  # apply cos to odd indices in the array; 2i+1
  angle_rads[, seq(2, dim(angle_rads)[2], by = 2)] <- cos(angle_rads[, seq(2, dim(angle_rads)[2], by = 2)])
  return(k_expand_dims(tfValFl32(angle_rads), axis = 1))
}




#         2.2.2. Mask                                 ####
create_padding_mask <- function(seq){
  seq <- tfValFl32(tf$math$equal(tfValFl32(seq), 0))
  return(seq[,tf$newaxis, tf$newaxis,])
}

create_look_ahead_mask <- function(size){
  mask <- 1 - tf$linalg$band_part(tf$ones(tfVal32(c(size, size))), tfVal32(-1), tfVal32(0))
}
#         2.2.3. Scaled dot-product attention         ####
scaled_dot_product_attention <- function(q, k, v, mask){
  # """Calculate the attention weights.
  # q, k, v must have matching leading dimensions.
  # k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  # The mask has different shapes depending on its type(padding or look ahead)
  # but it must be broadcastable for addition.
  # 
  # Args:
  #   q: query shape == (..., seq_len_q, depth)
  #   k: key shape == (..., seq_len_k, depth)
  #   v: value shape == (..., seq_len_v, depth_v)
  #   mask: Float tensor with shape broadcastable
  #       to (..., seq_len_q, seq_len_k). Defaults to None.
  # 
  # Returns:
  #   output, attention_weights
  # """
  matmul_qk <- tf$matmul(q, k, transpose_b = TRUE)  # (..., seq_len_q, seq_len_k)
  
  
  # scale matmul_qk
  dk <- tfValFl32(tf$shape(k)[-1])
  scaled_attention_logits <- matmul_qk / tf$math$sqrt(dk)
  
  # add the mask to the scaled tensor.
  if(!is.null(mask)){
    scaled_attention_logits <- scaled_attention_logits + (mask * -1e9)
  }
  
  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights <- tf$nn$softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  
  output <- tf$matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  
  return(list(output, attention_weights))
}
#         2.2.4. Multi-head attention                 ####
MultiHeadAttention <- Layer(classname = 'MultiHeadAttention',
                            initialize = function(d_model, num_heads) {
                              super()$`__init__`()
                              self$num_heads  = num_heads
                              self$d_model = d_model
                              self$depth = d_model %/% num_heads
                              self$wq = tf$keras$layers$Dense(d_model)
                              self$wk = tf$keras$layers$Dense(d_model)
                              self$wv = tf$keras$layers$Dense(d_model)
                              self$dense = tf$keras$layers$Dense(d_model)
                              # self$split_heads = function(x, batch_size){
                              #   # """Split the last dimension into (num_heads, depth).
                              #   #    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
                              #   # """
                              #   x = tf$reshape(x, tfVal32(c(batch_size, -1, self$num_heads, self$depth)))
                              # 
                              #   return(tf$transpose(x, perm=tfVal32(c(0, 2, 1, 3))))
                              # }
                            },
                            build = function(input_shape) {
                              # print(class(input_shape))
                              # input_shape1 <- input_shape
                              self$kernel <- self$add_weight(
                                name = "kernel",
                                shape = list(input_shape[[2]], self$d_model),
                                initializer = "uniform",
                                trainable = TRUE
                              )
                            },
                            
                            split_heads = function(x, batch_size){
                              # """Split the last dimension into (num_heads, depth).
                              #    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
                              # """
                              x = tf$reshape(x, tfVal32(c(batch_size, -1, self$num_heads, self$depth)))
                              
                              return(tf$transpose(x, perm=tfVal32(c(0, 2, 1, 3))))
                            },
                            call = function(v, k, q, mask){
                              batch_size <- tf$shape(q)[1]
                              
                              q <- self$wq(q)  # (batch_size, seq_len, d_model)
                              k <- self$wk(k)  # (batch_size, seq_len, d_model)
                              v <- self$wv(v)  # (batch_size, seq_len, d_model)
                              
                              q <- self$split_heads(q, as.numeric(batch_size))  # (batch_size, num_heads, seq_len_q, depth)
                              k <- self$split_heads(k, as.numeric(batch_size))  # (batch_size, num_heads, seq_len_k, depth)
                              v <- self$split_heads(v, as.numeric(batch_size))  # (batch_size, num_heads, seq_len_v, depth)
                              
                              # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
                              # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
                              c(scaled_attention, attention_weights) %<-% scaled_dot_product_attention(q, k, v, mask)
                              
                              scaled_attention <- tf$transpose(scaled_attention, perm=tfVal32(c(0, 2, 1, 3)))  # (batch_size, seq_len_q, num_heads, depth)
                              
                              concat_attention <- tf$reshape(scaled_attention, tfVal32(c(as.numeric(batch_size), -1, self$d_model)))  # (batch_size, seq_len_q, d_model)
                              
                              output <- self$dense(concat_attention)  # (batch_size, seq_len_q, d_model)
                              
                              return(list(output, attention_weights))
                            }
)

#         2.2.5. Point wise feed forward network      ####
point_wise_feed_forward_network <- function(d_model, dff){
  return(tf$keras$Sequential(list(
    tf$keras$layers$Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
    tf$keras$layers$Dense(d_model)  # (batch_size, seq_len, d_model)
  )))
}


#         2.2.6. Encoder layer                        ####
EncoderLayer <- Layer(classname = 'EncoderLayer',
                      initialize = function(d_model, 
                                            num_heads, 
                                            dff, 
                                            rate=0.1) {
                        super()$`__init__`()
                        self$mha = MultiHeadAttention(d_model = d_model, num_heads = num_heads)
                        self$ffn = point_wise_feed_forward_network(d_model, dff)
                        self$layernorm1 = tf$keras$layers$LayerNormalization(epsilon=1e-6)
                        self$layernorm2 = tf$keras$layers$LayerNormalization(epsilon=1e-6)
                        self$dropout1 = tf$keras$layers$Dropout(rate)
                        self$dropout2 = tf$keras$layers$Dropout(rate)
                      },
                      build = function(input_shape) {
                        # print(class(input_shape))
                        # input_shape1 <- input_shape
                        self$kernel <- self$add_weight(
                          name = "kernel",
                          shape = list(input_shape[[2]], self$d_model),
                          initializer = "uniform",
                          trainable = TRUE
                        )
                      },
                      
                      call = function(x, training, mask){
                        attn_output <- self$mha$call(x, k=x, q=x, mask = mask)[[1]] %>%   # (batch_size, input_seq_len, d_model)
                          self$dropout1(training = training)
                        # attn_output <- dropout1(attn_output, training = training)
                        out1 <- self$layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
                        
                        ffn_output <- self$ffn(out1) %>%   # (batch_size, input_seq_len, d_model)
                          self$dropout2(training=training)
                        
                        out2 <- self$layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
                        
                        return(out2)
                        
                      }
)

#         2.2.7. Decoder layer                        ####
DecoderLayer <- Layer(classname = 'DecoderLayer',
                      initialize = function(d_model, 
                                            num_heads, 
                                            dff, 
                                            rate=0.1) {
                        super()$`__init__`()
                        self$mha1 =  MultiHeadAttention(d_model = d_model, num_heads = num_heads)
                        self$mha2 =  MultiHeadAttention(d_model = d_model, num_heads = num_heads)
                        self$ffn = point_wise_feed_forward_network(d_model, dff)
                        self$layernorm1 = tf$keras$layers$LayerNormalization(epsilon=1e-6)
                        self$layernorm2 = tf$keras$layers$LayerNormalization(epsilon=1e-6)
                        self$layernorm3 = tf$keras$layers$LayerNormalization(epsilon=1e-6)
                        self$dropout1 = tf$keras$layers$Dropout(rate)
                        self$dropout2 = tf$keras$layers$Dropout(rate)
                        self$dropout3 = tf$keras$layers$Dropout(rate)
                      },
                      build = function(input_shape) {
                        # print(class(input_shape))
                        # input_shape1 <- input_shape
                        self$kernel <- self$add_weight(
                          name = "kernel",
                          shape = list(input_shape[[2]], self$d_model),
                          initializer = "uniform",
                          trainable = TRUE
                        )
                      },
                      
                      call = function(x, enc_output, training, look_ahead_mask, padding_mask){
                        # enc_output.shape == (batch_size, input_seq_len, d_model)
                        c(attn1, attn_weights_block1) %<-% self$mha1$call(x, k=x, q=x, mask = look_ahead_mask)  # (batch_size, target_seq_len, d_model)
                        attn1 <- self$dropout1(attn1, training=training)
                        
                        out1 <- self$layernorm1(attn1 + x)
                        
                        c(attn2, attn_weights_block2) %<-% self$mha2$call(
                          enc_output, k=enc_output, q=out1, mask = padding_mask)  # (batch_size, target_seq_len, d_model)
                        attn2 <- self$dropout2(attn2, training=training)
                        out2 <- self$layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
                        
                        ffn_output <- self$ffn(out2) %>%   # (batch_size, target_seq_len, d_model)
                          self$dropout3(training=training)
                        # ffn_output <- self$dropout3(ffn_output, training=training)
                        
                        out3 <- self$layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
                        
                        return(list(out3, attn_weights_block1, attn_weights_block2))
                        
                      }
)
#         2.2.8. Encoder                              ####
Encoder <- Layer(classname = 'Encoder',
                 initialize = function(num_layers,
                                       d_model,
                                       num_heads,
                                       dff,
                                       input_vocab_size,
                                       maximum_position_encoding,
                                       rate=0.1) {
                   super()$`__init__`()
                   
                   self$d_model = d_model
                   self$num_layers = num_layers
                   
                   self$embedding = layer_embedding(input_dim = input_vocab_size,
                                                    output_dim = d_model)
                   
                   
                   self$pos_encoding = positional_encoding(maximum_position_encoding, d_model)
                   
                   self$enc_layers = sapply(seq_len(self$num_layers), function(x)
                     EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, rate = rate)
                   )
                   
                   self$dropout = tf$keras$layers$Dropout(rate)
                   
                   
                 },
                 build = function(input_shape) {
                   # print(class(input_shape))
                   # input_shape1 <- input_shape
                   self$kernel <- self$add_weight(
                     name = "kernel",
                     shape = list(input_shape[[2]], self$d_model),
                     initializer = "uniform",
                     trainable = TRUE
                   )
                 },
                 
                 call = function(x, training, mask){
                   seq_length <- as.numeric(tf$shape(x)[2])
                   
                   # adding embedding and position encoding.
                   x <- self$embedding(x) * # (batch_size, input_seq_len, d_model)
                     tf$math$sqrt(tfValFl32(self$d_model)) +
                     self$pos_encoding[, seq_len(seq_length),] %>%
                     self$dropout(training=training)
                   
                   # x <- self$embedding(x) # (batch_size, input_seq_len, d_model)
                   # x <- x * tf$math$sqrt(tfValFl32(self$d_model))
                   # x <- x + self$pos_encoding[, seq_len(seq_length),]
                   # x <- self$dropout(x, training=training)
                   
                   # for(i in seq_len(length(self$enc_layers))){
                   for(i in 0:(self$num_layers-1)){
                     x <- self$enc_layers[[i]]$call(x, training, mask)
                   }
                   
                   return(x)
                   
                 }
)

#         2.2.9. Decoder                              ####
Decoder <- Layer(classname = 'Decoder',
                 initialize = function(num_layers,
                                       d_model,
                                       num_heads,
                                       dff,
                                       target_vocab_size,
                                       maximum_position_encoding,
                                       rate=0.1) {
                   super()$`__init__`()
                   
                   self$d_model = d_model
                   self$num_layers = num_layers
                   
                   self$embedding = layer_embedding(input_dim = target_vocab_size,
                                                    output_dim = d_model)
                   
                   
                   self$pos_encoding = positional_encoding(maximum_position_encoding, d_model)
                   
                   self$dec_layers = sapply(seq_len(self$num_layers), function(x) 
                     DecoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, rate = rate)
                   )
                   
                   self$dropout = tf$keras$layers$Dropout(rate)
                   
                   
                 },
                 build = function(input_shape) {
                   # print(class(input_shape))
                   # input_shape1 <- input_shape
                   self$kernel <- self$add_weight(
                     name = "kernel",
                     shape = list(input_shape[[2]], self$d_model),
                     initializer = "uniform",
                     trainable = TRUE
                   )
                 },
                 
                 call = function(x, enc_output, training, look_ahead_mask, padding_mask){
                   seq_length <- as.numeric(tf$shape(x)[2])
                   attention_weights <- list()
                   # adding embedding and position encoding.
                   x <- self$embedding(x) * # (batch_size, input_seq_len, d_model)
                     tf$math$sqrt(tfValFl32(self$d_model)) +
                     self$pos_encoding[, seq_len(seq_length),] %>%
                     self$dropout(training=training)
                   
                   # x <- self$embedding(x) # (batch_size, input_seq_len, d_model)
                   # x <- x * tf$math$sqrt(tfValFl32(self$d_model))
                   # x <- x + self$pos_encoding[, seq_len(seq_length),]
                   # x <- self$dropout(x, training=training)
                   
                   
                   # for(i in seq_len(length(self$enc_layers))){
                   for(i in 0:(self$num_layers-1)){
                     c(x, block1, block2) %<-% self$dec_layers[[i]]$call(x, enc_output, training,
                                                                         look_ahead_mask, padding_mask)
                     
                     
                     attention_weights[[paste0('decoder_layer_', i+1)]] <- list(
                       block1 = block1,
                       block2 = block2
                     )
                   }
                   
                   return(list(x, attention_weights))
                   
                 }
)


#         2.2.10. Transformer                         ####
Transformer <- function(num_layers, 
                        d_model, 
                        num_heads, 
                        dff, 
                        input_vocab_size,
                        target_vocab_size, 
                        pe_input, 
                        pe_target, 
                        rate=0.1,
                        name = 'Transformer'){
  keras_model_custom(name = name, function(self) {
    
    self$encoder = Encoder(num_layers = num_layers, 
                           d_model = d_model, 
                           num_heads = num_heads, 
                           dff = dff, 
                           input_vocab_size = input_vocab_size, 
                           maximum_position_encoding = pe_input, 
                           rate = rate)
    
    self$decoder = Decoder(num_layers = num_layers, 
                           d_model = d_model, 
                           num_heads = num_heads, 
                           dff = dff,
                           target_vocab_size = target_vocab_size, 
                           maximum_position_encoding = pe_target, 
                           rate = rate)
    
    self$final_layer = tf$keras$layers$Dense(target_vocab_size)
    
    self$create_masks = function(inp, tar){
      # Encoder padding mask
      enc_padding_mask <- create_padding_mask(inp)
      
      # Used in the 2nd attention block in the decoder.
      # This padding mask is used to mask the encoder outputs.
      dec_padding_mask <- create_padding_mask(inp)
      
      # Used in the 1st attention block in the decoder.
      # It is used to pad and mask future tokens in the input received by
      # the decoder.
      look_ahead_mask <- create_look_ahead_mask(tf$shape(tar)[2])
      dec_target_padding_mask <- create_padding_mask(tar)
      
      look_ahead_mask = tf$maximum(dec_target_padding_mask, look_ahead_mask)
      
      return(list(enc_padding_mask, look_ahead_mask, dec_padding_mask))
    }
    
    
    
    self$main <- function(inputs, training){
      # Keras models prefer if you pass all your inputs in the first argument
      c(inp, tar) %<-% inputs
      
      c(enc_padding_mask, look_ahead_mask, dec_padding_mask) %<-% self$create_masks(inp, tar)
      
      enc_output <- self$encoder$call(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
      
      # dec_output.shape == (batch_size, tar_seq_len, d_model)
      c(dec_output, attention_weights) %<-% self$decoder$call(
        tar, 
        enc_output, 
        training, 
        look_ahead_mask, 
        dec_padding_mask)
      
      final_output <- self$final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
      
      return(list(final_output, attention_weights))
      
      
    }
  })
}

transformer = Transformer(
  num_layers=num_layers,
  d_model=d_model,
  num_heads=num_heads,
  dff=dff,
  input_vocab_size=src_vocab_size,
  target_vocab_size=target_vocab_size,
  pe_input=pe_input,
  pe_target=pe_target,
  rate=dropout_rate)


transformer$load_weights(filepath = file.path(this.dir(), 'transformer_ing-dutch/transformer_ing-dutch'))

# # encoder$get_weights()
# # decoder$get_weights()
# encoder$load_weights(filepath = file.path(this.dir(), 'encoder_ing-germ/encoder_ing-germ'))
# decoder$load_weights(filepath = file.path(this.dir(), 'decoder_ing-germ/decoder_ing-germ'))

#    2.2. Translator                                  ####
translator <-
  function(sentence, need_attention = FALSE) {
    # sentence <- preprocess_sentence(sentence)
    # input <- sentence2digits(sentence, index_df = src_index)
    # input <-
    #   pad_sequences(list(input), maxlen = src_maxlen,  padding = "post") %>% 
    #   k_constant()
    
    input <- bpe_encode(model = model_src, 
                        x     = sentence, 
                        type  = "ids", 
                        bos   = TRUE, 
                        eos   = TRUE) %>% 
      pad_sequences(maxlen = src_maxlen,  padding = "post") %>% 
      k_constant()
    
    
    # Encoder and decoder padding mask 
    enc_padding_mask <- create_padding_mask(input)
    # Encoder part
    enc_output <- transformer$encoder$call(input, training = FALSE, enc_padding_mask) 
    
    result <- ""
    
    result_list <- list(
      as.data.table(model_target$vocabulary)['<BOS>', on = 'subword']$id
    )
    
    end_id <- as.data.table(model_target$vocabulary)['<EOS>', on = 'subword']$id
    
    dec_input <- k_expand_dims(result_list)
    
    # dec_input <- tf$TensorArray(dtype=tf$int64, size=tfVal64(0), dynamic_size=TRUE)
    # dec_input <- dec_input$write(tfVal64(0), tfVal64(word2index("<start>", target_index)))
    
    t <- 1
    while(TRUE){
      # output <- dec_input$stack() %>% k_expand_dims() %>% tf$transpose()
      # preds <- transformer$main(list(input, output), training=FALSE)[[1]]
      
      # dec_output.shape == (batch_size, tar_seq_len, d_model)
      look_ahead_mask <- create_look_ahead_mask(tf$shape(dec_input)[2])
      dec_target_padding_mask <- create_padding_mask(dec_input)
      
      look_ahead_mask <- tf$maximum(dec_target_padding_mask, look_ahead_mask)
      
      dec_output <- transformer$decoder$call(
        dec_input, 
        enc_output, 
        training = FALSE, 
        look_ahead_mask, 
        padding_mask = enc_padding_mask)[[1]]
      
      preds <- transformer$final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
      
      
      predictions <- preds[, -1, ]
      
      # Аз бир бирде болуўчу
      pred_idx <- tf$random$categorical(k_exp(predictions), num_samples = 1L)[1, 1] %>% as.double()
      
      # Тюз
      # pred_idx <- tf$argmax(predictions, axis=tfVal64(-1)) %>% as.double()
      
      
      # pred_word <- index2word(pred_idx, target_index)
      
      result_list[t+1] <- pred_idx
      t <- t + 1
      if (pred_idx == end_id) {
        break
      } else {
        # dec_input <- k_expand_dims(list(pred_idx))
        dec_input <- k_expand_dims(result_list) %>% tf$transpose()
        # dec_input <- dec_input$write(tfVal64(t), tfVal64(pred_idx))
      }
    }
    
    # result <- sapply(result_list, function(index) index2word(index, index_df = target_index)) %>% paste0(collapse = ' ') %>%  str_replace_all(pattern = '<start> | <stop>', replacement = '')
    
    
    
    if(need_attention){
      
      result <- 
        bpe_decode(model = model_target,
                   x = as.integer(unlist(result_list))) %>% 
        str_replace_all(pattern = '<BOS>|<EOS>', replacement = '') %>% 
        str_squish() %>% 
        str_replace_all("\\ ([:punct:])", "\\1")
      
      
      tokens <- pad_sequences(list(result_list), maxlen = target_maxlen,  padding = "post") %>% 
        k_constant()
      
      
      look_ahead_mask <- create_look_ahead_mask(tf$shape(tokens)[2])
      dec_target_padding_mask <- create_padding_mask(tokens)
      
      look_ahead_mask <- tf$maximum(dec_target_padding_mask, look_ahead_mask)
      
      attention_weights <- transformer$decoder$call(
        tokens, 
        enc_output, 
        training = FALSE, 
        look_ahead_mask, 
        padding_mask = enc_padding_mask)[[2]]
      
      # attention_weights <- transformer$main(list(input, tokens), training=FALSE)[[2]]
      
      # sentence <- str_replace_all(string = sentence, pattern = '<start> | <stop>', replacement = '')
      
      return(list(str_trim(result), sentence, attention_weights, tokens))
      
    } else {
      result <- 
        bpe_decode(model = model_target,
                   x = as.integer(unlist(result_list))) %>% 
        str_replace_all(pattern = '<BOS>|<EOS>', replacement = '') %>% 
        str_squish() %>% 
        str_replace_all("\\ ([:punct:])", "\\1")
      
      return(result)
    }
  }
#    2.3. Translation                                 ####
example_sentence <- 'What are you doing right now'#  train_sentences[[1]]

# tic()
translator(example_sentence)
# toc()
