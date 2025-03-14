market_values=("32" "40" "48" "56" "64" "72")
depth_values=("1" "2" "4")

for p1 in "${market_values[@]}"; do
  for p2 in "${depth_values[@]}"; do
    python train.py $p1 $p2
    echo "--------------------------------"  # Separator for clarity
  done
done
