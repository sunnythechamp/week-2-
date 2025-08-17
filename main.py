import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import PipelineException

# ---------------------------------------------------------
# Setup logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # ---------------------------------------------------------
        # Choose model (you can replace with any Hugging Face model)
        # ---------------------------------------------------------
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        logger.info(f"Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # ---------------------------------------------------------
        # Detect device (GPU if available, else CPU)
        # ---------------------------------------------------------
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")

        # ---------------------------------------------------------
        # Create pipeline
        # ---------------------------------------------------------
        classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        # ---------------------------------------------------------
        # Sample inputs (batching for performance)
        # ---------------------------------------------------------
        sentences = [
            "I love Hugging Face, it's amazing!",
            "This is the worst movie I've ever seen.",
            "Sunny days make me happy.",
            "I am not sure about this product."
        ]
        logger.info(f"Running inference on {len(sentences)} sentences...")

        results = classifier(sentences, batch_size=4)

        # ---------------------------------------------------------
        # Print results
        # ---------------------------------------------------------
        for sentence, result in zip(sentences, results):
            logger.info(f"Input: {sentence} | Prediction: {result}")

    except PipelineException as pe:
        logger.error(f"Pipeline error: {pe}")
    except Exception as e:
        logger.exception("Unexpected error occurred")

if __name__ == "__main__":
    main()
