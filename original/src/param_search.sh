market_values=("15" "20" "25" "32" "48" "64")
depth_values=("1" "3" "5")

for p1 in "${market_values[@]}"; do
  for p2 in "${depth_values[@]}"; do
    python train.py $p1 $p2
    echo "--------------------------------"  # Separator for clarity
  done
done
