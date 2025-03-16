market_values=("24" "32" "40" "48" "56" "64")
depth_values=("1")

for p1 in "${market_values[@]}"; do
  for p2 in "${depth_values[@]}"; do
    python train.py $p1 $p2
    echo "--------------------------------"  # Separator for clarity
  done
done
