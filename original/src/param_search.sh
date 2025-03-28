market_values=("40" "96" "124" "256")
depth_values=("2" "4" "6" "8")

for p1 in "${market_values[@]}"; do
  for p2 in "${depth_values[@]}"; do
    python train.py $p1 $p2
    echo "--------------------------------"  # Separator for clarity
  done
done
