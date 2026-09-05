import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, field_validator

# -------------------------------------------------------------------
# 1. Environment & Configuration Setup
# -------------------------------------------------------------------
load_dotenv()


# -------------------------------------------------------------------
# 2. Custom Domain Exceptions
# -------------------------------------------------------------------
class ModelInferenceError(Exception):
    """Raised when the AI model fails during prediction or processing."""
    pass


class ConfigurationError(Exception):
    """Raised when mandatory environment configurations are missing."""
    pass


# -------------------------------------------------------------------
# 3. Data Validation & Type Hints (Pydantic)
# -------------------------------------------------------------------
class ImagePayload(BaseModel):
    image_url: HttpUrl
    width: int = Field(gt=0, description="Width must be greater than 0")
    height: int = Field(gt=0, description="Height must be greater than 0")
    tags: Optional[list[str]] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def clean_and_validate_tags(cls, tags_list: Optional[list[str]]) -> list[str]:
        """Custom validator to sanitize tags list to lower-case."""
        if not tags_list:
            return []
        return [tag.strip().lower() for tag in tags_list if tag.strip()]


# -------------------------------------------------------------------
# 4. Object-Oriented Programming (OOP) for AI Systems
# -------------------------------------------------------------------
class BasePipeline:
    """Base abstract interface demonstrating OOP Inheritance & Encapsulation."""
    def __init__(self, model_name: str):
        self._model_name = model_name  # Encapsulated (protected) attribute

    def predict(self, payload: ImagePayload) -> dict:
        raise NotImplementedError("Subclasses must implement the predict method.")


class ModelPipeline(BasePipeline):
    """Production AI Pipeline handling model inference and validation."""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.__api_key = api_key  # Private attribute

    @classmethod
    def from_env(cls):
        """Factory method using @classmethod to instantiate pipeline from environment variables."""
        api_key = os.getenv("API_KEY")
        model_name = os.getenv("MODEL_NAME", "Default-AI-Model")

        if not api_key:
            raise ConfigurationError("Missing critical environment variable: API_KEY")

        return cls(model_name=model_name, api_key=api_key)

    @property
    def model_info(self) -> str:
        """Getter decorator for controlled attribute access."""
        return f"Model Architecture: {self._model_name}"

    def predict(self, payload: ImagePayload) -> dict:
        """Executes prediction on validated ImagePayload with robust error handling."""
        try:
            print(f"[LOG] Executing pipeline for payload URL: {payload.image_url}")

            if "fail" in str(payload.image_url):
                raise ModelInferenceError("Inference failed: Corrupted image stream.")

            result = {
                "status": "success",
                "model": self._model_name,
                "input_dimensions": f"{payload.width}x{payload.height}",
                "processed_tags": payload.tags,
                "confidence_score": 0.984
            }

        except ModelInferenceError as mie:
            print(f"[ERROR] Inference Exception Caught: {mie}")
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected Execution Failure: {e}")
            raise ModelInferenceError(f"Unhandled error during prediction: {e}")
        else:
            print("[SUCCESS] Prediction executed cleanly with zero errors.")
            return result
        finally:
            print("[CLEANUP] Releasing temporary GPU/RAM buffers...\n")

    def __repr__(self) -> str:
        """Dunder method for developer string representation."""
        return f"ModelPipeline(model='{self._model_name}', status='Active')"


# -------------------------------------------------------------------
# 5. Execution & Verification Workflow
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Week 1 Production Pipeline Integration Test ===\n")

    # Step A: Instantiate Pipeline via Classmethod Factory
    try:
        pipeline = ModelPipeline.from_env()
        print(f"Initialized Pipeline: {repr(pipeline)}")
        print(f"Pipeline Info: {pipeline.model_info}\n")
    except ConfigurationError as ce:
        print(f"Initialization Failed: {ce}")
        exit(1)

    # Step B: Valid Payload Execution
    print("--- Case 1: Valid Image Payload Processing ---")
    raw_data = {
        "image_url": "https://example.com/assets/sample_image.png",
        "width": 1920,
        "height": 1080,
        "tags": [" ObjectDetection ", " LANDSCAPE ", ""]
    }

    try:
        payload = ImagePayload(**raw_data)
        prediction = pipeline.predict(payload)
        print(f"Prediction Result: {prediction}\n")
        print(f"Serialized Payload JSON:\n{payload.model_dump_json(indent=2)}\n")
    except Exception as e:
        print(f"Payload validation or execution failed: {e}\n")

    # Step C: Trigger Error Handling via Invalid Pipeline Input
    print("--- Case 2: Custom ModelInferenceError Trigger ---")
    fail_data = {
        "image_url": "https://example.com/fail_stream.png",
        "width": 800,
        "height": 600,
        "tags": ["test"]
    }

    try:
        fail_payload = ImagePayload(**fail_data)
        pipeline.predict(fail_payload)
    except ModelInferenceError:
        print("[HANDLED] Custom ModelInferenceError successfully caught at caller level.")