import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 로컬 이미지 경로
image_path = "/home/su/idle_ws/test_images/red_blue_green.png"
image = Image.open(image_path).convert("RGB")

# 4GB GPU면 크기 살짝 줄이는 게 안전
image.thumbnail((640, 640))

# 찾고 싶은 텍스트
text_labels = "a red block. a blue block. a green block. a basket."

print("device:", DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_ID
).to(DEVICE)

model.eval()

inputs = processor(
    images=image,
    text=text_labels,
    return_tensors="pt"
)

inputs = {
    k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
    for k, v in inputs.items()
}

with torch.inference_mode():
    outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs["input_ids"],
    threshold=0.3,
    text_threshold=0.2,
    target_sizes=target_sizes
)[0]

print("\nDetections:")
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(x, 2) for x in box.tolist()]
    print(f"label={label}, box={box}")

draw = ImageDraw.Draw(image)

for label, box in zip(results["labels"], results["boxes"]):
    x1, y1, x2, y2 = box.tolist()
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1), f"{label}", fill="black")

save_path = "/home/su/idle_ws/test_images/dino_result1.png"
image.save(save_path)
print(f"\nSaved result image to: {save_path}")