"""
위성영상 분석 프롬프트 템플릿
한국어/영어 이중 지원 (모델은 영어 프롬프트로 추론 → 한국어 응답 요청)
"""

from config.settings import settings


def _lang_suffix() -> str:
    if settings.DEFAULT_LANGUAGE == "ko":
        return " Please respond in Korean (한국어로 답변해주세요)."
    return ""


# ────────────────────────────────────────────────────────────────
# 1. 단일 영상 설명 (Describe)
# ────────────────────────────────────────────────────────────────

DESCRIBE_GENERAL = """You are an expert remote sensing analyst. 
Analyze this satellite image and provide a comprehensive description including:
1. Overall scene type and land use/cover
2. Key features and objects visible (buildings, roads, vegetation, water bodies, etc.)
3. Approximate scale and spatial relationships
4. Image quality and any notable characteristics (shadows, resolution, artifacts)
5. Potential geographic region if identifiable
Be specific and technical in your analysis.{lang}"""

DESCRIBE_WITH_CONTEXT = """You are an expert remote sensing analyst.
Given that this satellite image is from {context}, analyze it and describe:
1. Scene content relevant to {context}
2. Notable features and their significance
3. Any anomalies or points of interest
4. Confidence level in your assessment
{lang}"""


# ────────────────────────────────────────────────────────────────
# 2. 영상 비교 (Compare)
# ────────────────────────────────────────────────────────────────

COMPARE_TEMPORAL = """You are an expert remote sensing analyst specializing in change detection.
You are given TWO satellite images of the SAME area taken at DIFFERENT times.
Image 1: Before / Earlier image
Image 2: After / Later image

Analyze and describe:
1. CHANGES detected between the two images:
   - Land use/cover changes
   - New constructions or demolitions
   - Vegetation changes (growth, deforestation, seasonal)
   - Water body changes
   - Infrastructure changes
2. UNCHANGED areas and features
3. Severity and extent of changes (estimate percentage of area affected)
4. Possible causes of the changes (natural, human, disaster-related)
5. Temporal confidence (how certain are you these changes are real vs artifacts)
{lang}"""

COMPARE_GENERAL = """You are an expert remote sensing analyst.
Compare these two satellite images and describe:
1. Key similarities between the two images
2. Key differences in content, features, and characteristics
3. Image quality comparison (resolution, clarity, coverage)
4. What each image shows distinctively
{lang}"""


# ────────────────────────────────────────────────────────────────
# 3. 구름 분석 (Cloud Coverage)
# ────────────────────────────────────────────────────────────────

CLOUD_ANALYSIS = """You are an expert remote sensing image quality analyst.
Analyze the cloud coverage in this satellite image and provide:

1. CLOUD COVERAGE ESTIMATE:
   - Overall cloud coverage percentage (0-100%)
   - Thick cloud coverage (opaque, blocking ground): ___%
   - Thin cloud/haze coverage (semi-transparent): ___%
   - Cloud-free area: ___%

2. SPATIAL DISTRIBUTION:
   - Which parts of the image are most affected? (e.g., NW corner, central region)
   - Is the coverage uniform or patchy?

3. CLOUD TYPES (if identifiable):
   - Cumulus, Stratus, Cirrus, or mixed
   - Cloud shadow presence

4. USABILITY ASSESSMENT:
   - Overall usability for analysis: [Excellent/Good/Fair/Poor/Unusable]
   - Recommended for: [specify use cases this image is still suitable for]
   - Critical areas obscured: [list important features hidden by clouds]

5. QUALITY SCORE: [1-10] where 10 = completely cloud-free
{lang}"""


# ────────────────────────────────────────────────────────────────
# 4. 세그멘테이션 / 특정 지점 분석 (Segment & Point)
# ────────────────────────────────────────────────────────────────

POINT_DESCRIPTION = """You are an expert remote sensing analyst.
Focus on the HIGHLIGHTED/MARKED REGION in this satellite image.
The region of interest is marked with a red rectangle/highlight.

Describe this specific region:
1. What features are present in this region?
2. Land use/cover type
3. Dimensions and scale (if estimable)
4. Surrounding context and relationship to nearby features
5. Any notable characteristics or anomalies
6. Classification confidence
{lang}"""

COORDINATE_DESCRIPTION = """You are an expert remote sensing analyst.
In this satellite image, focus on the area around pixel coordinates ({x}, {y}).
The image dimensions are {width}x{height} pixels.
This corresponds to approximately {lat_rel} from the top and {lon_rel} from the left of the image.

Describe what is at/near this location:
1. Primary feature or object at this point
2. Immediate surrounding features (within ~50px radius)
3. Land use/cover classification
4. Relationship to larger structures or patterns
5. Confidence in identification
{lang}"""

OBJECT_DETECTION = """You are an expert remote sensing analyst specializing in object detection.
Detect and describe ALL instances of {target_object} in this satellite image.

For each detected instance provide:
1. Approximate location (describe position: NW/NE/SW/SE quadrant, center, etc.)
2. Estimated size/scale
3. Confidence level (High/Medium/Low)
4. Distinguishing characteristics

Also provide:
- Total count of detected {target_object}
- Detection confidence summary
- Any objects that might be {target_object} but you are uncertain about
{lang}"""

CHANGE_DETECTION_POINT = """You are an expert remote sensing change detection analyst.
Compare the SAME LOCATION in these two satellite images.
Focus on the area around the marked/specified region.

Analyze:
1. What has CHANGED at this specific location?
2. Nature of change: [Natural/Construction/Demolition/Damage/Other]
3. Estimated time of change (if discernible)
4. Magnitude of change: [Minor/Moderate/Major/Complete transformation]
5. Surrounding area changes for context
6. Confidence in change detection: [High/Medium/Low]
{lang}"""

SEGMENTATION_REQUEST = """You are an expert remote sensing analyst.
Perform semantic segmentation analysis of this satellite image.

Identify and describe ALL distinct land cover/use categories present:
1. For each category:
   - Category name (e.g., Urban/Residential, Forest, Agricultural, Water, Bare soil, etc.)
   - Approximate percentage of total image area
   - Spatial distribution description
   - Confidence of classification

2. Dominant land cover (largest category)
3. Any boundary areas or mixed pixels between categories
4. Overall landscape characterization
{lang}"""


def get_describe_prompt(context: str = None) -> str:
    lang = _lang_suffix()
    if context:
        return DESCRIBE_WITH_CONTEXT.format(context=context, lang=lang)
    return DESCRIBE_GENERAL.format(lang=lang)


def get_compare_prompt(mode: str = "temporal") -> str:
    lang = _lang_suffix()
    if mode == "temporal":
        return COMPARE_TEMPORAL.format(lang=lang)
    return COMPARE_GENERAL.format(lang=lang)


def get_cloud_prompt() -> str:
    return CLOUD_ANALYSIS.format(lang=_lang_suffix())


def get_point_prompt(x: int, y: int, width: int, height: int,
                     lat: float = None, lon: float = None) -> str:
    lang = _lang_suffix()
    lat_rel = f"{y/height*100:.0f}%"
    lon_rel = f"{x/width*100:.0f}%"
    return COORDINATE_DESCRIPTION.format(
        x=x, y=y, width=width, height=height,
        lat_rel=lat_rel, lon_rel=lon_rel, lang=lang
    )


def get_object_detection_prompt(target: str) -> str:
    return OBJECT_DETECTION.format(
        target_object=target, lang=_lang_suffix()
    )


def get_change_point_prompt() -> str:
    return CHANGE_DETECTION_POINT.format(lang=_lang_suffix())


def get_segmentation_prompt() -> str:
    return SEGMENTATION_REQUEST.format(lang=_lang_suffix())
