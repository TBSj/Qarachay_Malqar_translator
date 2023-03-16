# 1. Libraries and consts             ####
library(keras)
library(tensorflow)
library(tfdatasets)
library(readr)
library(dplyr)
library(data.table)
library(this.path)
library(tokenizers.bpe)
library(stringr)
library(reticulate)
library(telegram.bot.dt)
# virtualenv_create("r-reticulate", python = install_python())
# install_keras(envname = "r-reticulate")
# library(tfautograph)
# reticulate::py_install('tensorflow_hub', pip = TRUE)
# reticulate::py_install('tensorflow-text', pip = TRUE)
# reticulate::py_install('keras-bert', pip = TRUE)
# reticulate::py_install('transformers', pip = TRUE)
# hub = reticulate::import('tensorflow_hub')
# tf_text = reticulate::import('tensorflow_text')
# transformers = reticulate::import('transformers', as = 'ts)
# k_bert = import('keras_bert')
# token_dict = k_bert$load_vocabulary(vocab_path)
# tokenizer = k_bert$Tokenizer(token_dict)

# gpu_options = tf$compat$v1$GPUOptions(per_process_gpu_memory_fraction=0.98)
# sess = tf$compat$v1$Session(config = tf$compat$v1$ConfigProto(gpu_options=gpu_options))
# tf$test$is_gpu_available()
# tf$device('/cpu:0')
# PATH_TF_HUB <- 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/2'

# FILEPATH_SOURCE_PREPARED <- str_replace(this.dir(), strsplit(this.dir(), '/')[[1]] %>% last(), "Source_prepared")
FILEPATH_SOURCE_PREPARED <- "1.Data"

FILE_TEXT <- file.path(FILEPATH_SOURCE_PREPARED, "Bary_bir_to_model.csv")
# FILE_TEXT <- file.path(FILEPATH_SOURCE_PREPARED, "Bary_besh_to_model.csv")

SRC_LANG <- 'rus'
TRG_LANG <- 'krc'

# SRC_LANG <- 'krc'
# TRG_LANG <- 'rus'

# MODEL_PATH_SRC_BPE    <- file.path(this.dir(), 'BPE', paste0("youtokentome_", SRC_LANG, ".bpe"))
# MODEL_PATH_TARGET_BPE <- file.path(this.dir(), 'BPE', paste0("youtokentome_", TRG_LANG, ".bpe"))
MODEL_DIR <- str_replace(this.dir(), 'Qarachay_Malqar_translator/2.R', "Models")
MODEL_PATH_SRC_BPE    <- file.path(MODEL_DIR, 'BPE', paste0("youtokentome_", SRC_LANG, ".bpe"))
MODEL_PATH_TARGET_BPE <- file.path(MODEL_DIR, 'BPE', paste0("youtokentome_", TRG_LANG, ".bpe"))

TRANSFORMER_NAME <- paste0('transformer_', SRC_LANG, '_to_',TRG_LANG, '.h5')

# TRANSFORMER_FILE <- file.path(this.dir(), 'model', TRANSFORMER_NAME)
TRANSFORMER_FILE <- file.path(MODEL_DIR, TRANSFORMER_NAME)

TRANSFORMER_NAME_ALL <- paste0('transformer_', SRC_LANG, '_to_',TRG_LANG, '_all.h5')

TRANSFORMER_FILE_ALL <- file.path(MODEL_DIR, TRANSFORMER_NAME_ALL)

# 2. Preparing                        ####
#    2.1. Download dataset            ####
text_pairs <- FILE_TEXT %>%
  fread()
  # .[seq(15000)] # for examples

str(text_pairs[sample(nrow(text_pairs), 1), ])

# src_vocab_size_for_bpe    <- round(strsplit(text_pairs[[SRC_LANG]],    ' ') %>% unlist() %>% uniqueN() * 0.7)
# target_vocab_size_for_bpe <- round(strsplit(text_pairs[[TRG_LANG]], ' ') %>% unlist() %>% uniqueN() * 0.7)

#    2.2. BPE                         ####
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

sequence_length <- max(trg_maxlen, src_maxlen) # 152

src_matrix <-
  pad_sequences(src_diglist, maxlen = sequence_length,  padding = "post")

trg_matrix <-
  pad_sequences(trg_diglist, maxlen = sequence_length + 1, padding = "post")

#    2.3. Train-test-split            ####
num_test_samples <- 10
num_val_samples <- round(0.2 * nrow(text_pairs))

num_train_samples <- nrow(text_pairs) - num_val_samples - num_test_samples

pair_group <- sample(base::c(
  rep("train", num_train_samples),
  rep("test", num_test_samples),
  rep("val", num_val_samples)
))

test_pairs <- text_pairs[pair_group == "test", ]

x_train <- src_matrix[pair_group == "train",]
y_train <- trg_matrix[pair_group == "train",]

x_valid <- src_matrix[pair_group == "val",]
y_valid <- trg_matrix[pair_group == "val",]

#    2.4. Hyperparameters / variables ####

src_vocab_size <- model_src_bpe$vocab_size # 46962
trg_vocab_size <- model_trg_bpe$vocab_size # 40520


dropout_rate <- 0.4 # Дропаут
embed_dim <- 256  # Количество нейронов в слое (или глубина внимания) (Эм иги 512, компьютерим тарталмайды)
dense_dim <- 1024 #  Расстояние позиций (токенов) для слоя позиционно-зависимого прямого распространения (Эм иги 2048, компьютерим тарталмайды)
num_heads <- 8 # Кол-во голов внимания (8)
# num_layers <- 4 # Кол-во слоёв (6)


buffer_size <- nrow(x_train)
learning_rate  <-  1e-3 # 1e-4
epoches <- 5
ep_stop <- 2
batch_size <- 16 # 64
regul <- regularizer_l1_l2(l1 = learning_rate, l2 = learning_rate)
#    2.5. Preparing matrix tf         ####
# Слой векторизации можно вызывать как с пакетными, так и с непакетными данными. Здесь мы применяем векторизацию перед пакетной обработкой данных
format_pair <- function(pair) {
  src_p <- pair$src %>% as_tensor(dtype = 'int64')
  trg_p <- pair$trg %>% as_tensor(dtype = 'int64')
  
  # Опускаем последний токен в испанском предложении, чтобы входные данные и цели имели одинаковую длину. [NA:–2] удаляет последний элемент тензора
  inputs <- list()  
  inputs[[SRC_LANG]] = src_p
  inputs[[TRG_LANG]] = trg_p[NA:-2]
  # [2:NA] удаляет первый элемент тензора
  targets <- trg_p[2:NA]
  
  # Целевое испанское предложение на один шаг впереди.
  # Оба имеют одинаковую длину (20 слов)
  list(inputs, targets)
}

train_ds <- tensor_slices_dataset(keras_array(list(src = x_train, trg = y_train))) %>% 
  dataset_map(format_pair) %>% 
  dataset_shuffle(buffer_size = buffer_size) %>% 
  dataset_batch(batch_size)  
  # dataset_prefetch(16)

val_ds <- tensor_slices_dataset(keras_array(list(src = x_valid, trg = y_valid))) %>% 
  dataset_map(format_pair) %>% 
  dataset_shuffle(buffer_size = buffer_size) %>% 
  dataset_batch(batch_size)  
  # dataset_prefetch(16)


# c(inputs, targets) %<-% iter_next(as_iterator(train_ds))
# c(inputs_v, targets_v) %<-% iter_next(as_iterator(val_ds))
# str(inputs)

#    2.6. Bot                         ####

MobiDickMessage <- function(text = 'юйренибди', chat_id = '428576415', token = "5195278336:AAEn7oQyXox7CQFzNpQHnjNqqC1KKBU3MgY"){

  bot <- Bot(token=token)

  mess <- bot$sendMessage(chat_id = chat_id,
                          text = text,
                          parse_mode = "Markdown"
  )

}

# 3. Preparing model                  ####
#    3.1. PositionalEmbedding         ####
layer_positional_embedding <- new_layer_class(
  classname = "PositionalEmbedding",
  initialize = function(sequence_length, # Недостатком позиционного встраивания является то, что длина последовательности должна быть известна заранее
                        input_dim, output_dim, ...) {
    super$initialize(...)
    self$token_embeddings <- # Подготовка layer_embedding() для индексов токенов
      layer_embedding(input_dim = input_dim,
                      output_dim = output_dim)
    
    # Подготовка layer_embedding() для позиций токенов
    self$position_embeddings <-
      layer_embedding(input_dim = sequence_length,
                      output_dim = output_dim)
    self$sequence_length <- sequence_length
    self$input_dim <- input_dim
    self$output_dim <- output_dim
  },
  call = function(inputs) {
    len <- tf$shape(inputs)[-1] # tf$shape(inputs)[–1] вырезает последний элемент формы (tf$shape() возвращает форму в виде тензора)
    positions <-
      tf$range(start = 0L, limit = len, delta = 1L) # tf$range() похож на seq() в R, создает целочисленную последовательность: [0, 1, 2, …, limit – 1]
    embedded_tokens <- self$token_embeddings(inputs)
    embedded_positions <- self$position_embeddings(positions)
    embedded_tokens + embedded_positions # Складываем векторы встраивания
  },
  
  # Как и layer_embedding(), этот слой должен уметь генерировать маску, чтобы мы могли игнорировать заполнение нулями во входных данных. Метод calculate_mask() будет автоматически вызван фреймворком, и маска распространится на следующий уровень
  compute_mask = function(inputs, mask = NULL) {
    inputs != 0
  },
  # Сериализация для сохранения модели
  get_config = function() {
    config <- super$get_config()
    for(name in base::c("output_dim", "sequence_length", "input_dim"))
      config[[name]] <- self[[name]]
    config
  }
)
#    3.2. Transformer Encoder layer   ####
layer_transformer_encoder <- new_layer_class(
  classname = "TransformerEncoderLayer",
  # initialize = function(embed_dim, dense_dim, num_heads, dropout = 0.3, ...) {
  initialize = function(embed_dim, dense_dim, num_heads, ...) {
    super$initialize(...)
    self$embed_dim <- embed_dim # Размер векторов входных токенов
    self$dense_dim <- dense_dim # Размер внутреннего слоя Dense
    self$num_heads <- num_heads # Количество голов внимания
    self$attention <-
      layer_multi_head_attention(num_heads = num_heads,
                                 key_dim = embed_dim)
    self$dense_proj <- keras_model_sequential() %>%
      layer_dense(dense_dim, activation = "relu") %>%
      # layer_dropout(dropout) %>%
      layer_dropout(dropout_rate) %>%
      layer_dense(embed_dim)
    self$layernorm_1 <- layer_layer_normalization()
    self$layernorm_2 <- layer_layer_normalization()
  },
  call = function(inputs, mask = NULL) {
    # Маска, которая будет сгенерирована слоем встраивания, будет двумерной, но слой внимания ожидает, что она будет трех- или четырехмерной, поэтому мы расширяем ее размерность
    if (!is.null(mask))
      mask <- mask[, tf$newaxis, ]
    inputs %>%
      { self$attention(., ., attention_mask = mask) + . } %>% # Добавляем остаточную связь к выходу слоя dense_proj()
      self$layernorm_1() %>%
      { self$dense_proj(.) + . } %>% # Добавляем остаточную связь к выходу слоя внимания
      self$layernorm_2()
  },
  # Выполняем сериализацию, благодаря чему можно сохранить модель
  get_config = function() {
    config <- super$get_config()
    for(name in base::c("embed_dim", "num_heads", "dense_dim"))
      config[[name]] <- self[[name]]
    config
  }
)

# preprocessor = hub$KerasLayer(
# "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

# layer_transformer_encoder_tf_hub <- hub$KerasLayer(PATH_TF_HUB, trainable = TRUE)
#    3.3. Transformer Decoder layer   ####
layer_transformer_decoder <- new_layer_class(
  classname = "TransformerDecoderLayer",
  # initialize = function(embed_dim, dense_dim, num_heads, dropout = 0.3, ...) {
  initialize = function(embed_dim, dense_dim, num_heads, ...) {
    super$initialize(...)
    self$embed_dim <- embed_dim
    self$dense_dim <- dense_dim
    self$num_heads <- num_heads
    self$attention_1 <- layer_multi_head_attention(num_heads = num_heads,
                                                   key_dim = embed_dim)
    self$attention_2 <- layer_multi_head_attention(num_heads = num_heads,
                                                   key_dim = embed_dim)
    self$dense_proj <- keras_model_sequential() %>%
      layer_dense(dense_dim, activation = "relu") %>%
      # layer_dropout(dropout) %>%
      layer_dropout(dropout_rate) %>%
      layer_dense(embed_dim)
    
    self$layernorm_1 <- layer_layer_normalization()
    self$layernorm_2 <- layer_layer_normalization()
    self$layernorm_3 <- layer_layer_normalization()
    self$supports_masking <- TRUE # Этот атрибут гарантирует, что слой будет распространять свою входную маску на свои выходные данные; маскировка в Keras явно включена. Если вы передаете маску слою, который не реализует compute_mask() и не предоставляет этот атрибут supports_masking, это ошибка
  },
  get_config = function() {
    config <- super$get_config()
    for (name in base::c("embed_dim", "num_heads", "dense_dim"))
      config[[name]] <- self[[name]]
    config
  },
  get_causal_attention_mask = function(inputs) {
    c(batch_size, sequence_length, .) %<-% # Третья ось — encoding_length; мы не используем ее здесь
      tf$unstack(tf$shape(inputs))
    
    x <- tf$range(sequence_length) # Целочисленная последовательность [0, 1, 2, … sequence_length–1]
    i <- x[, tf$newaxis]
    j <- x[tf$newaxis, ]
    # mask представляет собой квадратную матрицу формы (sequence_length, sequence_length), с 1 в нижнем треугольнике и 0 в остальных местах. Например, если sequence_length равно 4
    mask <- tf$cast(i >= j, "int32") # Используем изменение размерности тензора в нашей операции >=. Приводим dtype bool к int32
    
    
    tf$tile(mask[tf$newaxis, , ],
            tf$stack(base::c(batch_size, 1L, 1L))) # Добавляем размер пакета в маску, а затем размещаем (rep()) вдоль оси пакета batch_size раз. Возвращенный тензор имеет форму (batch_size, sequence_length, sequence_length)
  },
  
  call = function(inputs, encoder_outputs, mask = NULL) {
    
    causal_mask <- self$get_causal_attention_mask(inputs) # Получаем каузальную маску
    
    # Маска, предоставляемая в вызове, является маской заполнения (она описывает места заполнения в целевой последовательности)
    if (is.null(mask))
      mask <- causal_mask
    else
      mask %<>% { tf$minimum(tf$cast(.[, tf$newaxis, ], "int32"),
                             causal_mask) } # Объединяем маску заполнения с каузальной маской
    
    inputs %>%
      { self$attention_1(query = ., value = ., key = .,
                         attention_mask = causal_mask) + . } %>% # Передайте причинную маску первому слою внимания, который применяет самовнимание к целевой последовательности
      self$layernorm_1() %>% # Выход attention_1() с добавленным остатком передается в layernorm_1()
      { self$attention_2(query = .,
                         value = encoder_outputs, # Используем encoder_outputs, предоставленные в вызове, в качестве значения и ключа для warning_2()
                         key = encoder_outputs,
                         attention_mask = mask) + . } %>% # Передаем комбинированную маску второму слою внимания, который связывает исходную последовательность с целевой последовательностью
      self$layernorm_2() %>% # Выход attention_2() с добавленным остатком передается в layernorm_2()
      { self$dense_proj(.) + . } %>%
      self$layernorm_3() # Выход dense_proj() с добавленным остатком передается в layernorm_3()
  })


# sequence_length
#    3.4. Transformer Encoder         ####
# transformer_encoder <- new_layer_class(
#   classname = "TransformerEncoder",
#   initialize = function(embed_dim, dense_dim, num_heads, num_layers, ...) {
#     super$initialize(...)
#     # self$embed_dim <- embed_dim # Размер векторов входных токенов
#     # self$dense_dim <- dense_dim # Размер внутреннего слоя Dense
#     # self$num_heads <- num_heads # Количество голов внимания
#     self$num_layers <- num_layers # Количество слоёв
#     self$enc_layer <- sapply(seq(num_layers), function(x)
#       # layer_transformer_encoder(NULL, embed_dim, dense_dim, num_heads, dropout)
#       layer_transformer_encoder(NULL, embed_dim, dense_dim, num_heads)
#     )
#   },
#   call = function(inputs, mask = NULL) {
#     # Маска, которая будет сгенерирована слоем встраивания, будет двумерной, но слой внимания ожидает, что она будет трех- или четырехмерной, поэтому мы расширяем ее размерность
#     x <- inputs
# 
#     # for(i in 0:(self$num_layers  - 1)){
#     for(i in seq(self$num_layers)){
#       x <- self$enc_layer[[i]](x)
#     }
# 
#     return(x)
# 
#   },
#   # Выполняем сериализацию, благодаря чему можно сохранить модель
#   get_config = function() {
#     config <- super$get_config()
#     for(name in c("embed_dim", "num_heads", "dense_dim", 'num_layers'))
#       config[[name]] <- self[[name]]
#     config
#   }
# )
#    3.5. Transformer Decoder         ####
# transformer_decoder <- new_layer_class(
#   classname = "TransformerDecoder",
#   # initialize = function(embed_dim, dense_dim, num_heads, num_layers, dropout = 0.3, ...) {
#   initialize = function(embed_dim, dense_dim, num_heads, num_layers, ...) {
#     super$initialize(...)
#     self$num_layers <- num_layers # Количество слоёв
#     self$dec_layer <- sapply(seq_len(self$num_layers), function(x)
#       # layer_transformer_decoder(NULL, embed_dim, dense_dim, num_heads, dropout)
#       layer_transformer_decoder(NULL, embed_dim, dense_dim, num_heads)
#     )
#   },
#   get_config = function() {
#     config <- super$get_config()
#     for (name in c("embed_dim", "num_heads", "dense_dim", 'num_layers'))
#       config[[name]] <- self[[name]]
#     config
#   },
# 
#   call = function(inputs, encoder_outputs, mask = NULL) {
#     x <- inputs
# 
#     # for(i in 0:(self$num_layers-1)){
#     for(i in seq(self$num_layers)){
#       x <- self$dec_layer[[i]](x, encoder_outputs)
#     }
#     return(x)
# 
#   })




#    3.6. Transformer                 ####
#    3.6.1. First type Transformer    ####

# encoder_inputs <- layer_input(shape(NA), dtype = "int64", name = SRC_LANG)
encoder_inputs <- layer_input(shape(NA), dtype = "float64", name = SRC_LANG)
# Кодируем исходную последовательность
encoder_outputs <- encoder_inputs %>%
  layer_positional_embedding(sequence_length, src_vocab_size, embed_dim) %>%
  # layer_transformer_encoder(embed_dim, dense_dim, num_heads, dropout = dropout_rate)
  layer_transformer_encoder(embed_dim, dense_dim, num_heads) %>% 
  # layer_transformer_encoder(embed_dim, dense_dim, num_heads) %>% 
  layer_transformer_encoder(embed_dim, dense_dim, num_heads) %>% 
  layer_transformer_encoder(embed_dim, dense_dim, num_heads) 

# Передаем NULL в качестве первого аргумента, чтобы экземпляр слоя создавался и возвращался напрямую, ни с чем не смешиваясь
# transformer_decoder1 <-
#   # layer_transformer_decoder(NULL, embed_dim, dense_dim, num_heads, dropout = dropout_rate)
#   layer_transformer_decoder(NULL, embed_dim, dense_dim, num_heads)

# decoder_inputs <- layer_input(shape(NA), dtype = "int64", name = TRG_LANG)
decoder_inputs <- layer_input(shape(NA), dtype = "float64", name = TRG_LANG)
decoder_outputs <- decoder_inputs %>%
  layer_positional_embedding(sequence_length, trg_vocab_size, embed_dim) %>%

  # transformer_decoder(encoder_outputs) %>% # Кодируем целевое предложение и объединяем его с закодированным исходным предложением
  layer_transformer_decoder(NULL, embed_dim, dense_dim, num_heads)(., encoder_outputs) %>% 
  layer_transformer_decoder(NULL, embed_dim, dense_dim, num_heads)(., encoder_outputs) %>% 
  # layer_transformer_decoder(NULL, embed_dim, dense_dim, num_heads)(., encoder_outputs) %>% 
  layer_transformer_decoder(NULL, embed_dim, dense_dim, num_heads)(., encoder_outputs) %>% 
  layer_dropout(dropout_rate) %>%
  layer_dense(trg_vocab_size, activation = "softmax") # Предсказываем слово для каждой выходной позиции

#    3.6.2. Second type Transformer   ####

# encoder_inputs <- layer_input(shape(NA), dtype = "int64", name = SRC_LANG)
# # Кодируем исходную последовательность
# encoder_outputs <- encoder_inputs %>%
#   layer_positional_embedding(sequence_length, src_vocab_size, embed_dim) %>%
#   # transformer_encoder(embed_dim, dense_dim, num_heads, num_layers = num_layers, dropout = dropout_rate)
#   transformer_encoder(embed_dim, dense_dim, num_heads, num_layers = num_layers)
# 
# # Передаем NULL в качестве первого аргумента, чтобы экземпляр слоя создавался и возвращался напрямую, ни с чем не смешиваясь
# decoder <-
#   # transformer_decoder(NULL, embed_dim, dense_dim, num_heads, num_layers = num_layers, dropout = dropout_rate)
#   transformer_decoder(NULL, embed_dim, dense_dim, num_heads, num_layers = num_layers)
# 
# decoder_inputs <- layer_input(shape(NA), dtype = "int64", name = TRG_LANG)
# decoder_outputs <- decoder_inputs %>%
#   layer_positional_embedding(sequence_length, trg_vocab_size, embed_dim) %>%
# 
#   decoder(encoder_outputs) %>% # Кодируем целевое предложение и объединяем его с закодированным исходным предложением
#   layer_dropout(dropout_rate) %>%
#   layer_dense(trg_vocab_size, activation = "softmax") # Предсказываем слово для каждой выходной позиции

#    3.6.3. Transformer               ####

transformer <- keras_model(list(encoder_inputs, decoder_inputs),
                           decoder_outputs)


# 4. Fit                              ####
# c(decay_steps, warmup_steps) %<-% k_bert$calc_train_steps(
#   y_train %>% length(), # 3332952
#   batch_size=batch_size,
#   epochs=epoches
# )
# 
# opt <- k_bert$AdamWarmup(decay_steps=decay_steps,
# warmup_steps=warmup_steps, learning_rate = learning_rate)

transformer <- load_model_hdf5(TRANSFORMER_FILE,
                                custom_objects =
                                  list(PositionalEmbedding = layer_positional_embedding,
                                       TransformerEncoderLayer = layer_transformer_encoder,
                                       TransformerDecoderLayer = layer_transformer_decoder
                                       # AdamWarmup = opt
                                       )
                               )



transformer %>%
  compile(optimizer =  keras$optimizers$Adam(learning_rate = learning_rate), #opt,#
          loss = 'sparse_categorical_crossentropy',
          metrics = 'acc')


callbacks_list <- list(
  # 
  callback_early_stopping(
    monitor = "val_loss",
    patience = ep_stop), # Прерываем обучение, когда точность проверки перестанет улучшаться в течение двух эпох
  
  callback_model_checkpoint(
    filepath = TRANSFORMER_FILE,
    monitor = "val_loss",
    save_best_only = TRUE)
)



transformer %>%
  fit(train_ds, epochs = epoches, validation_data = val_ds, callbacks = callbacks_list) # 30
  # fit(train_ds, epochs = epoches, validation_data = val_ds) # 30
  # fit(train_ds, epochs = 1, validation_data = val_ds, callbacks = callbacks_list) # 30
MobiDickMessage(paste0(Sys.time(), ' заманда юйренибди'))

# 5. Save model                       ####
save_model_hdf5(transformer, filepath = TRANSFORMER_FILE_ALL)

transformer1 <- load_model_hdf5(TRANSFORMER_FILE_ALL,
                                custom_objects = 
                                  list(PositionalEmbedding = layer_positional_embedding,
                                       TransformerEncoderLayer = layer_transformer_encoder,
                                       TransformerDecoderLayer = layer_transformer_decoder
                                       # AdamWarmup = opt
                                       ))

# 6. Translation                      ####

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
    str_replace_all('\\bсыпат', 'сыфат') %>%
    str_replace_all('\\bСыпат', 'Сыфат') %>%
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
# tf_function

decodeSequence <- (function(input_row) {
  
  # withr::local_options(tensorflow.extract.style = "python")
  
  end_id <- as.data.table(model_trg_bpe$vocabulary)['<EOS>', on = 'subword']$id 
  decoded_row <- as.data.table(model_trg_bpe$vocabulary)['<BOS>', on = 'subword']$id %>% 
    as_tensor(shape = c(1, 1))

  for (i in seq(sequence_length)) {
  # for (i in seq(20)) {
    
    decoded_row <- pad_sequences(decoded_row, maxlen = sequence_length, padding = 'post') %>% as_tensor(dtype = 'int64')
    
    next_token_predictions <-
      transformer(list(input_row,
                       decoded_row))
    
    # greedy 0.02 s
    sampled_token_index <- tf$argmax(next_token_predictions[1, i, ]) %>% as.numeric()
    # beam search, 3 s
    # k_ctc_decode(next_token_predictions[, , ], input_length = as_tensor(sequence_length, shape = c(1)), greedy = FALSE, beam_width = 3)
    # sampled_token_index <- tf$random$categorical(k_exp(next_token_predictions[, i, ]), num_samples = 1L)[1, 1] %>% as.numeric()
    
    
    decoded_row <- list(c(as.numeric(decoded_row)[seq(i)], sampled_token_index))
    
    if (sampled_token_index == end_id){
      break
    }
  }
  # decoded_row
  unlist(decoded_row) %>% 
    as.integer()
})


translator <-
  function(sentence) {

    input <- bpe_encode(model = model_src_bpe, 
                        x     = sentence, 
                        type  = "ids", 
                        bos   = TRUE, 
                        eos   = TRUE) %>% 
      pad_sequences(maxlen = sequence_length,  padding = "post") %>% 
      as_tensor(dtype = 'int64')
    
    output <- decodeSequence(input)

      result <- 
        bpe_decode(model = model_trg_bpe,
                   x = output) %>% 
        str_replace_all(pattern = '<BOS>|<EOS>|<PAD>', replacement = '') %>% 
        str_squish() %>% 
        fromModel()
      
      return(result)
    
  }




# Проверка
# for (i in sample.int(nrow(test_pairs), 3)) {
for (i in seq(3)) {
  # c(input_sentence, correct_translation) %<-% test_pairs[i, ]
  input_sentence <- test_pairs[i, ][[SRC_LANG]]
  correct_translation <- test_pairs[i, ][[TRG_LANG]]
  cat(input_sentence, "\n")
  cat(input_sentence %>%
        translator(), "\n", correct_translation, "\n")
  cat("-\n")
}

