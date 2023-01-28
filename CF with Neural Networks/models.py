from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Add, Activation, Lambda, BatchNormalization, Concatenate, Dropout, Input, Embedding, Dot, Reshape, Dense, Flatten, Multiply

def EmbeddingDotBias(n_factors=64, learning_rate=0.001, batch_norm=True, use_bias=True, loss="binary_crossentropy"):
    n_factors = 64

    # Define input layers for users and movies
    user_input = Input(shape=(1,))
    anime_input = Input(shape=(1,))
    
    # Define embedding layers for users and animes
    user_embedding = Embedding(n_users, n_factors, name='user_embedding')(user_input)
    anime_embedding = Embedding(n_animes, n_factors, name='anime_embedding')(anime_input)

    # Define bias layers for users and animes
    user_bias = Embedding(n_users, 1, name='user_bias')(user_input)
    anime_bias = Embedding(n_animes, 1, name='anime_bias')(anime_input)

    # Flatten the embedding and bias layers
    user_embedding = Flatten()(user_embedding)
    anime_embedding = Flatten()(anime_embedding)
    user_bias = Flatten()(user_bias)
    anime_bias = Flatten()(anime_bias)

    # Compute dot product of user and anime embeddings
    dot_product = Dot(axes=1)([user_embedding, anime_embedding])

    # Add user and anime biases to the dot product
    if use_bias:
        output = Add()([dot_product, user_bias, anime_bias])
    else:
        output = dot_product
    
    # Output 
    if batch_norm:
        output = BatchNormalization()(output)
    output = Activation('sigmoid')(add)
#     output = output * (max_rating-min_rating) + min_rating 

    # Define the model
    model = Model(inputs=[user_input, anime_input], outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=["mae", "mse", "binary_crossentropy"])
    
    model_config = {
        "model_name": "embedding_dot_bias",
        "learning_rate": learning_rate,
        "loss": loss,
        "n_factors": n_factors,
        "batch_norm": batch_norm,
        "use_biad": use_bias
    }
    
    return model, model_config
