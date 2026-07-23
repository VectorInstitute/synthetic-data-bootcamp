## Quality evaluation for small Number of rare samples:

#### Formally Evaluating Synthetic Data Quality for Rare Conditions

This is one of the hardest open problems in the field, because most standard metrics were designed for abundant real data. When your real reference set is 100–500 images of a rare condition, nearly every classical metric breaks down in a non-obvious way. Here is a structured framework.

  


**The Core Problem:** Metric Breakdown at Small N

FID (Fréchet Inception Distance) — the industry default — requires at minimum 2,048–10,000 real images for a statistically stable estimate. At N=200 real images, FID variance is so large that the confidence interval swallows any meaningful difference between good and bad generators. The same applies to IS (Inception Score) and precision/recall in feature space.

  


This means you cannot rely on distribution-level generative metrics alone and must use a layered evaluation stack with different methods compensating for each other's blind spots.

  


##### Layer 1 — Visual Fidelity (per-image, N-tolerant)

These work on small sets because they operate per-image or per-pair rather than estimating a distribution.

  


**CLIP-Score / CLIP-alignment:** Measures whether the generated image matches its conditioning prompt. Since you always have the prompt you used, this is computable for every single generated image. A low CLIP score means the generator ignored the rare-condition description.

  


Reference: ++[github.com/jmhessel/clipscore](http://github.com/jmhessel/clipscore)++  
  


**DINO / DINOv2 patch-level similarity:** Compare patch-level feature maps between a synthetic image and its most similar real neighbor. DINOv2 features are much more spatially faithful than CLIP and catch structural implausibility (a limb bending the wrong way, a rail track with impossible geometry) that CLIP misses.

  


Reference: ++[github.com/facebookresearch/dinov2](http://github.com/facebookresearch/dinov2)++  
  


**LPIPS (Learned Perceptual Image Patch Similarity):** When you have paired inputs (e.g., you started from a real image via img2img), LPIPS measures perceptual distortion — useful for checking that structure was preserved and only the rare attribute was injected.

  


Reference: ++[github.com/richzhang/PerceptualSimilarity](http://github.com/richzhang/PerceptualSimilarity)++  
  


**KID (Kernel Inception Distance) over your small real set:** KID has an unbiased estimator, unlike FID, making it usable at small N (even N=50). It is noisier but directionally reliable.

  


Reference: Included in torchmetrics and clean-fid  
  


##### Layer 2 — Semantic / Condition Faithfulness (the rare-condition-specific gate)

This is the layer that standard generative evaluation completely ignores, and it is the most important one for your use case. The question is not just "does this look real?" but "does this actually show the rare condition you asked for?"

  


**VLM-as-Judge (structured prompting):** Run a VLM (Qwen2.5-VL-7B or InternVL3-8B) on every synthetic image with a structured multi-question prompt:

  


Q1: Does this image show [rare condition]? (yes/no + confidence)

Q2: Where in the image is the [rare condition] located? (bounding description)

Q3: Is the [rare condition] physically plausible given the scene? (yes/no)

Q4: Does the annotation mask cover the correct region? (yes/no)

Report the VLM-acceptance rate (fraction passing all four checks) as a first-class metric. A generator that scores 90% visual fidelity but 40% VLM-acceptance is useless for the rare-condition task.

  


**CLIP-based condition presence score:** Compute CLIP_sim(image, "a photo showing [rare condition]") - CLIP_sim(image, "a photo showing [normal condition]"). A positive delta confirms the rare attribute is detectably present. This is cheap, interpretable, and does not need a reference set.

  


**Attribute classifier agreement:** If you have even 50 real rare-condition images, train a tiny linear probe on frozen DINOv2 or CLIP features to classify "rare vs. normal." Then run your synthetic images through it. A synthetic image that your own small-data classifier rejects is not fooling your downstream model either. This is a domain-expert proxy at almost zero compute cost.

  


##### Layer 3 — Downstream Utility (the only metric that ultimately matters)

Train on Synthetic, Test on Real (TSTR) This is the gold standard. Fine-tune a small detector or classifier on synthetic only or real + synthetic, then evaluate on your held-out real rare-condition images. Report the metric gap vs. real only. If TSTR ≈ TRTR (Train on Real, Test on Real), your synthetic data has captured what the model actually needs to learn.

  


**The critical nuance for rare conditions:** report the metric separately on the rare class, not as an aggregate. Aggregate mAP can look fine while tail-class AP is zero — which is the failure mode you are trying to avoid.

  


**Calibration shift:** After fine-tuning on real + synthetic, measure whether model confidence on real rare-condition images is calibrated (ECE — Expected Calibration Error). Poorly calibrated synthetic data produces overconfident predictions on the rare class. This is a silent failure mode that TSTR accuracy alone misses.

  


Reference: ++[github.com/gpleiss/temperature_scaling](http://github.com/gpleiss/temperature_scaling)++  
  


**Precision-Recall on the rare class specifically:** For detection tasks, AP at IoU 0.5 on the rare class before and after adding synthetic data. For classification, per-class recall. These are the numbers that matter operationally in every target industry.

  


##### Layer 4 — Diversity and Coverage (avoiding mode collapse on rare conditions)

A generator that produces 1,000 copies of nearly the same rare scene is *worse than useless* — it will cause the model to overfit to one visual variant of the rare condition.

  


**Vendi Score:** A diversity metric based on matrix rank that measures the effective number of distinct samples in a set. Works at small N and does not require a reference distribution. Particularly useful for detecting mode collapse in small rare-condition sets.

  


Reference: ++[github.com/vertaix/Vendi-Score](http://github.com/vertaix/Vendi-Score)++  
  


**Intra-set DINO cosine similarity:** Compute the mean pairwise cosine similarity of DINOv2 features across your synthetic set. A high mean (> 0.85) signals mode collapse. This is a cheap proxy for Vendi Score.

  


**Condition-attribute coverage grid:** Since rare conditions combine multiple factors (e.g., weather × defect type × lighting × viewpoint), build an explicit grid of the combinations you want to cover and track what fraction your generator has produced at least one credible example of. A generator with high average fidelity but low grid coverage will fail in deployment when the specific combination it missed shows up in the real world. This is a manual but invaluable exercise.

  


##### Layer 5 — Annotation Quality (specific to detection / segmentation tasks)

Mask IoU between auto-annotation and VLM description After Grounded-SAM-2 auto-annotates, have the VLM describe where the rare condition is ("upper-left corner, roughly 15% of image area"). Convert that description into an approximate bounding box and compute IoU with the SAM mask. Anything < 0.3 IoU gets rejected.

  


**Annotation consistency across augmentations:** Apply a small geometric augmentation (flip, crop, rotate) to a synthetic image and check that the annotation transforms consistently. This catches SAM masks that latched onto a background artifact rather than the actual rare-condition object.

  


## Practical Evaluation Protocol Summary

Given the constraint of having very few real rare-condition images, a realistic protocol for this PoC looks like:

  


**Per-image gates (run on every generated image, discard failures):** CLIP-condition score, VLM-acceptance rate, annotation IoU check. These are your filters before anything enters the training pool.

  


**Set-level quality (run once on the curated pool):** KID against your small real set, intra-set DINO similarity (diversity check), Vendi Score, condition coverage grid.

  


**Downstream utility (run once after fine-tuning):** TSTR rare-class AP/recall, calibration ECE on the rare class.

  


**Human spot-check (N=50–100):** One domain expert reviews a random sample and rates physical plausibility on a 1–3 scale. This is irreplaceable for catching domain-specific failure modes that no automated metric catches — e.g., ice that forms in a physically impossible location, or a chest X-ray where rib geometry is subtly wrong.

  


The honest conclusion from the 2024–2026 literature is that **no single metric is trustworthy** for rare-condition synthetic data, and the only evaluation that definitively answers "did this help?" is **TSTR rare-class AP**. Everything else is a proxy that helps you filter bad generations before paying the cost of fine-tuning.