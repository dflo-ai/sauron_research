echo "Running full run..."
echo "===================="
echo "Processing demo.mp4..."
echo "----------------------"
python scripts/run_evaluation.py data/videos/demo.mp4 --output ./outputs
echo "Processing demo_k1.mp4..."
echo "-------------------------"
python scripts/run_evaluation.py data/videos/demo_k1.mp4 --output ./outputs
echo "Processing demo2.mp4..."
echo "----------------------"
python scripts/run_evaluation.py data/videos/demo2.mp4 --output ./outputs
echo "Processing cctv.mp4..."
echo "----------------------"
python scripts/run_evaluation.py data/videos/cctv.mp4 --output ./outputs