from SUT2_training import (
    TrainConfig,
    set_global_seed,
    build_data_generators,
    build_binary_cnn,
    print_generator_info,
)


def main():
    config = TrainConfig(
        data_root=r"C:\Users\akhiat\Desktop\Hackathon\ASAB\DATA\ScenAIro",
        image_size=(100, 100),
        batch_size=16,
        epochs=1,
        seed=42,
    )

    set_global_seed(config.seed)

    train_generator, validation_generator, test_generator = build_data_generators(config)
    print_generator_info(train_generator, validation_generator, test_generator)

    model = build_binary_cnn(
        input_shape=(config.image_size[0], config.image_size[1], 3)
    )

    print("\nModel built successfully.")

    x_batch, y_batch = next(train_generator)
    print("\nOne batch check:")
    print("x_batch shape:", x_batch.shape)
    print("y_batch shape:", y_batch.shape)
    print("x_batch min:", x_batch.min())
    print("x_batch max:", x_batch.max())
    print("labels in batch:", y_batch[:10])

    print("\nRunning one mini training step...")
    model.fit(train_generator, epochs=1, steps_per_epoch=1, verbose=1)

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()