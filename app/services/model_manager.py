"""
모델 매니저
지원 백엔드: GeoChat, Qwen2-VL, InternVL2, Mock (개발용)
에어갭 환경: 로컬 경로 우선, HuggingFace Hub 차선
"""

import logging
from typing import Optional

from config.settings import ModelBackend, settings

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.is_loaded = False
        self.backend = settings.MODEL_BACKEND

    async def load(self):
        """백엔드에 맞는 모델 로드"""
        if self.backend == ModelBackend.MOCK:
            logger.info("Mock 백엔드 활성화 (GPU/모델 불필요)")
            self.is_loaded = True
            return

        import torch
        model_path = settings.ACTIVE_MODEL_PATH
        logger.info(f"모델 로딩: {model_path} (backend={self.backend})")

        if self.backend == ModelBackend.GEOCHAT:
            await self._load_geochat(model_path)
        elif self.backend == ModelBackend.QWEN2_VL:
            await self._load_qwen2_vl(model_path)
        elif self.backend == ModelBackend.INTERNVL2:
            await self._load_internvl2(model_path)
        else:
            await self._load_llava(model_path)

        self.is_loaded = True

    async def _load_geochat(self, model_path: str):
        """GeoChat (LLaVA-1.5 기반 RS 특화 모델) 로드"""
        from transformers import AutoTokenizer
        # GeoChat은 LLaVA 아키텍처를 따름
        # 실제 배포 시: pip install git+https://github.com/mbzuai-oryx/GeoChat.git
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path

            model_name = get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device_map=settings.DEVICE,
                torch_dtype=torch.float16,
            )
            logger.info("GeoChat 로드 완료")
        except ImportError:
            logger.warning("GeoChat 패키지 없음 → LLaVA 호환 모드로 전환")
            await self._load_llava(model_path)

    async def _load_qwen2_vl(self, model_path: str):
        """Qwen2-VL 로드 (fine-tuning 후 권장 모델)"""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        dtype = torch.float16 if settings.TORCH_DTYPE == "float16" else torch.bfloat16

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=settings.DEVICE,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        logger.info("Qwen2-VL 로드 완료")

    async def _load_internvl2(self, model_path: str):
        """InternVL2 로드"""
        from transformers import AutoModel, AutoTokenizer
        import torchvision.transforms as T

        dtype = torch.float16 if settings.TORCH_DTYPE == "float16" else torch.bfloat16

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=settings.DEVICE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        logger.info("InternVL2 로드 완료")

    async def _load_llava(self, model_path: str):
        """LLaVA-1.5 fallback 로드"""
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        dtype = torch.float16 if settings.TORCH_DTYPE == "float16" else torch.float32

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=settings.DEVICE,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        logger.info("LLaVA fallback 로드 완료")

    async def unload(self):
        """메모리 해제"""
        if self.backend == ModelBackend.MOCK:
            self.is_loaded = False
            logger.info("Mock 백엔드 종료")
            return
        import torch
        if self.model is not None:
            del self.model
            del self.processor
            if self.tokenizer:
                del self.tokenizer
            torch.cuda.empty_cache()
            self.is_loaded = False
            logger.info("모델 메모리 해제 완료")

    def infer(self, prompt: str, images: list, **kwargs) -> str:
        """
        통합 추론 인터페이스
        백엔드별로 올바른 추론 메서드 호출
        """
        if self.backend == ModelBackend.MOCK:
            return self._infer_mock(prompt, images, **kwargs)
        if self.backend in (ModelBackend.GEOCHAT, ModelBackend.LLAVA):
            return self._infer_llava_style(prompt, images, **kwargs)
        elif self.backend == ModelBackend.QWEN2_VL:
            return self._infer_qwen2_vl(prompt, images, **kwargs)
        elif self.backend == ModelBackend.INTERNVL2:
            return self._infer_internvl2(prompt, images, **kwargs)
        raise ValueError(f"지원하지 않는 백엔드: {self.backend}")

    def _infer_mock(self, prompt: str, images: list, **kwargs) -> str:
        """개발/테스트용 mock 추론 - 프롬프트 키워드 기반 응답 생성"""
        p = prompt.lower()
        if "change" in p and "detect" in p:
            return (
                "Change Detection Analysis:\n"
                "Comparing the two time periods at the specified location:\n"
                "- Before: Open agricultural field with sparse vegetation\n"
                "- After: New residential development with 12-15 buildings\n"
                "- Key changes: Land conversion from agricultural to urban use\n"
                "- Change severity: High (complete land use transformation)"
            )
        if "detect" in p or "find all" in p:
            return (
                "Object Detection Results:\n"
                "Total objects detected: 8 instances\n"
                "- 3 objects located in the northern quadrant\n"
                "- 2 objects in the central area near road intersection\n"
                "- 3 objects in the southern region along the waterfront\n"
                "Confidence: moderate to high"
            )
        if "segment" in p or "land cover" in p or "land use" in p:
            return (
                "Land cover classification results:\n"
                "- Urban/Built-up: 35% (central and eastern portions)\n"
                "- Forest/Vegetation: 28% (northwestern hills)\n"
                "- Agricultural: 20% (southern plains)\n"
                "- Water bodies: 10% (river running east-west)\n"
                "- Bare soil/Exposed: 7% (construction sites)"
            )
        if "cloud" in p:
            return (
                "Cloud Analysis Results:\n"
                "- Overall cloud coverage: 42.3%\n"
                "- Cloud types: Cumulus (dominant), scattered cirrus\n"
                "- Clear regions: Southern and eastern portions\n"
                "- Shadow impact: 8% additional area affected\n"
                "- Quality Score: 6/10\n"
                "- Recommendation: Partially usable for analysis"
            )
        if "compare" in p:
            return (
                "Image Comparison Analysis:\n"
                "- Spatial extent: Both images cover similar geographic area\n"
                "- Notable differences: Vegetation density change in western sector\n"
                "- Urban expansion visible in southeastern quadrant\n"
                "- Water level variation detected in central reservoir"
            )
        if "point" in p or "pixel" in p or "location" in p:
            return (
                "Point Analysis:\n"
                "The specified location shows a mixed urban-commercial area.\n"
                "- Primary feature: Multi-story commercial building\n"
                "- Surrounding context: Paved roads, parking areas\n"
                "- Vegetation: Sparse street trees along the road"
            )
        # default: describe
        return (
            "This satellite image shows a dense urban area with a mix of "
            "residential and commercial buildings. The area features a grid-like "
            "road network with moderate vegetation coverage along main streets. "
            "A river or waterway is visible in the eastern portion of the image. "
            "The overall image quality is good with high spatial resolution, "
            "suitable for detailed urban analysis and building detection tasks."
        )

    def _infer_llava_style(self, prompt: str, images: list, **kwargs) -> str:
        """LLaVA / GeoChat 추론"""
        import torch
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        full_prompt = f"<image>\n{prompt}"
        inputs = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        image_tensors = process_images(
            images, self.processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs,
                images=image_tensors,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                do_sample=settings.TEMPERATURE > 0,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # 프롬프트 부분 제거
        return output.split("ASSISTANT:")[-1].strip()

    def _infer_qwen2_vl(self, prompt: str, images: list, **kwargs) -> str:
        """Qwen2-VL 추론"""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                do_sample=settings.TEMPERATURE > 0,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]

    def _infer_internvl2(self, prompt: str, images: list, **kwargs) -> str:
        """InternVL2 추론"""
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        pixel_values_list = [
            transform(img).unsqueeze(0).to(
                self.model.device,
                dtype=torch.float16 if settings.TORCH_DTYPE == "float16" else torch.bfloat16
            )
            for img in images
        ]

        if len(pixel_values_list) > 1:
            pixel_values = torch.cat(pixel_values_list, dim=0)
            num_patches_list = [1] * len(images)
        else:
            pixel_values = pixel_values_list[0]
            num_patches_list = None

        generation_config = {
            "max_new_tokens": settings.MAX_NEW_TOKENS,
            "do_sample": settings.TEMPERATURE > 0,
            "temperature": settings.TEMPERATURE,
            "top_p": settings.TOP_P,
        }

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
            num_patches_list=num_patches_list,
        )
        return response
