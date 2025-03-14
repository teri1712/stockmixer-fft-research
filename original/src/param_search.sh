market_values=("30" "35" "40" "45" "50")
depth_values=("4" "6" "8" "10" "12")

for p1 in "${market_values[@]}"; do
  for p2 in "${depth_values[@]}"; do
    python train.py $p1 $p2
    echo "--------------------------------"  # Separator for clarity
  done
done
