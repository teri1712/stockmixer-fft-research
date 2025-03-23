market_values=("48" "64" "96" "128" "164" "256")
depth_values=("8" "10" "12")

for p1 in "${market_values[@]}"; do
  for p2 in "${depth_values[@]}"; do
    python train.py $p1 $p2
    echo "--------------------------------"  # Separator for clarity
  done
done
