for i in 41 #28 31 37 38 44 61 62 63
do
python main_QL.py -epochs 10000 -runs 1 \
    -L_num 200 -alpha 0.8 -gamma 0.8 \
    -epsilon 0.3 -question 2 \
    -is_QL -is_not_fermi -seed ${i}
done
