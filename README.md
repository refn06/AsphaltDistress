🚧 Asphalt Pavement Distress Detection 
This application is a web-based system for detecting asphalt pavement distress using YOLOv9c (Compact) deep learning architecture.

Users can upload an image of a road surface (including drone-view perspectives), and the system will automatically identify and localize different types of pavement damage with high precision.

🛠️ Detected Distress Categories:
- Longitudinal Crack (D00): Cracks that run parallel to the road's centerline.
- Transverse Crack (D10): Cracks that run perpendicular to the road's centerline.
- Alligator Crack (D20): Interconnecting cracks forming a pattern similar to alligator skin.
- Pothole (D40): Bowl-shaped depressions in the pavement surface.
- Repair Area: Previously patched or repaired sections of the road.

🚀 Features:
- Advanced Model: Utilizes YOLOv9c with Programmable Gradient Information (PGI) for better feature extraction.
- Multi-Perspective: Optimized for both eye-level (vehicle camera) and top-down (drone/UAV) imagery.
- Civil Engineering Focused: Tailored for infrastructure monitoring and academic research.

📊 Model Performance:
The current model achieved an overall mAP50 of 0.68, with the highest accuracy in detecting Longitudinal Cracks (81.6%).
