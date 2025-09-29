from src.train import train_model

if __name__ == "__main__":
    train_dir = "archive/train"
    test_dir = "archive/test"

    model, history = train_model(train_dir, test_dir, resume=True)
    print("Training Completed. Model saved in saved_models folder.")
