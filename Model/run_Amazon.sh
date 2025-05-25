set -o errexit

# SOURCE_NODES=("product" "product" "product" "product" "product" "product" "product" "product" "product" "product" "product" "product" "product" "product" "product" "product" "product" "product")
# TARGET_NODES=("review" "review" "review" "review" "review" "review" "review" "review" "review" "review" "review" "review" "review" "review" "review" "review" "review" "review")
# SOURCE_TASKS=('fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake')
# TARGET_TASKS=('fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake' 'fake')
# BATCH_SIZES=(128 128 128 128 128 128 64 64 64 64 64 64 32 32 32 32 32 32)
# TRAIN_BATCHES=(19 19 19 19 19 19 38 38 38 38 38 38 76 76 76 76 76 76)
# TEST_BATCHES=(8 8 8 8 8 8 16 16 16 16 16 16 32 32 32 32 32 32)
# TRAIN_BATCHES2=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
# TEST_BATCHES2=(200 200 200 200 200 200 400 400 400 400 400 400 800 800 800 800 800 800)
# MATCHING_HOPSS=(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
# DROPOUTS=(0.1 0.1 0.3 0.3 0.5 0.5 0.1 0.1 0.3 0.3 0.5 0.5 0.1 0.1 0.3 0.3 0.5 0.5)
# LR=(0.0001 0.001 0.0001 0.001 0.0001 0.001 0.0001 0.001 0.0001 0.001 0.0001 0.001 0.0001 0.001 0.0001 0.001 0.0001 0.001)



# SOURCE_NODES=("product")
# TARGET_NODES=("review")
# SOURCE_TASKS=('fake')
# TARGET_TASKS=('fake')
# BATCH_SIZES=(128)
# TRAIN_BATCHES=(19)
# TEST_BATCHES=(8)
# TRAIN_BATCHES2=(300)
# TEST_BATCHES2=(7100)
# MATCHING_HOPSS=(1)
# DROPOUTS=(0.0)

# length=${#SOURCE_NODES[@]}

# for ((i=0;i<$length;i++))
# do
#     python main.py --dataset "graph_Amazon/" \
#          --source_node "${SOURCE_NODES[$i]}" --target_node "${TARGET_NODES[$i]}" \
#          --source_task "${SOURCE_TASKS[$i]}" --target_task "${TARGET_TASKS[$i]}" \
#          --train_batch ${TRAIN_BATCHES[$i]} --test_batch ${TEST_BATCHES[$i]} \
#          --train_batch2 ${TRAIN_BATCHES2[$i]} --test_batch2 ${TEST_BATCHES2[$i]} \
#          --matching_hops ${MATCHING_HOPSS[$i]} --no_matching_loss \
#          --dropout ${DROPOUTS[$i]} --batch_size ${BATCH_SIZES[$i]}
# done


######################################### 
# Ali's Version

SOURCE_NODES=("product")
TARGET_NODES=("reviewer")
SOURCE_TASKS=('fake')
TARGET_TASKS=('fake')
BATCH_SIZES=(64)
TRAIN_BATCHES=(38)
TEST_BATCHES=(16)
TRAIN_BATCHES2=(15292)
TEST_BATCHES2=(6554)
MATCHING_HOPSS=(1)
DROPOUTS=(0.01)
LR=(0.001)


# SOURCE_NODES=("product")
# TARGET_NODES=("reviewer")
# SOURCE_TASKS=('fake')
# TARGET_TASKS=('fake')
# BATCH_SIZES=(64)
# TRAIN_BATCHES=(38)
# TEST_BATCHES=(16)
# TRAIN_BATCHES2=(14015)
# TEST_BATCHES2=(6007)
# MATCHING_HOPSS=(1)
# DROPOUTS=(0.0)
# LR=(0.001)


length1=${#SOURCE_NODES[@]}
length2=${#DROPOUTS[@]}
length3=${#LR[@]}


for ((i=0;i<$length1;i++))
do
    for ((j=0;j<$length2;j++))
    do
        for ((k=0;k<$length3;k++))
        do
            python main.py --dataset "graph_Amazon_Ali/" \
                 --source_node "${SOURCE_NODES[$i]}" --target_node "${TARGET_NODES[$i]}" \
                 --source_task "${SOURCE_TASKS[$i]}" --target_task "${TARGET_TASKS[$i]}" \
                 --train_batch ${TRAIN_BATCHES[$i]} --test_batch ${TEST_BATCHES[$i]} \
                 --train_batch2 ${TRAIN_BATCHES2[$i]} --test_batch2 ${TEST_BATCHES2[$i]} \
                 --matching_hops ${MATCHING_HOPSS[$i]} --no_matching_loss \
                 --dropout ${DROPOUTS[$j]} --lr ${LR[$k]} --batch_size ${BATCH_SIZES[$i]}
        done
    done
done








# BATCH_SIZES=(64)
# TRAIN_BATCHES=(38)
# TEST_BATCHES=(16)




## computer science
#SOURCE_NODES=("paper" "author" "author" "venue" "paper" "author" "author" "venue")
#TARGET_NODES=("author" "paper" "venue" "author" "author" "paper" "venue" "author")
#SOURCE_TASKS=("L1" "L1" "L1" "L1" "L2" "L2" "L2" "L2")
#TARGET_TASKS=("L1" "L1" "L1" "L1" "L2" "L2" "L2" "L2")
#TRAIN_BATCHES=(200 200 200 60 400 400 400 60)
#TEST_BATCHES=(50 50 50 40 50 50 50 40)
#TRAIN_BATCHES2=(2 2 2 2 10 10 10 10)
#TEST_BATCHES2=(50 50 40 50 50 50 40 50)
#MATCHING_HOPSS=(1 1 2 2 1 1 2 2)
#
#length=${#SOURCE_NODES[@]}
#
#for ((i=0;i<=$length;i++))
#do
#    python main.py --dataset "graph_CS/" \
#         --source_node "${SOURCE_NODES[$i]}" --target_node "${TARGET_NODES[$i]}" \
#         --source_task "${SOURCE_TASKS[$i]}" --target_task "${TARGET_TASKS[$i]}" \
#         --train_batch ${TRAIN_BATCHES[$i]} --test_batch ${TEST_BATCHES[$i]} \
#         --train_batch2 ${TRAIN_BATCHES2[$i]} --test_batch2 ${TEST_BATCHES2[$i]} \
#         --matching_hops ${MATCHING_HOPSS[$i]} --no_matching_loss
#done


## computer network
#SOURCE_NODES=("paper" "author" "author" "venue")
#TARGET_NODES=("author" "paper" "venue" "author")
#SOURCE_TASKS=("L2" "L2" "L2" "L2")
#TARGET_TASKS=("L2" "L2" "L2" "L2")
#TRAIN_BATCHES=(250 250 250 25)
#TEST_BATCHES=(50 50 50 5)
#TRAIN_BATCHES2=(10 10 10 10)
#TEST_BATCHES2=(50 50 5 50)
#MATCHING_HOPSS=(1 1 2 2)
#
#length=${#SOURCE_NODES[@]}
#
#for ((i=0;i<=$length;i++))
#do
#    python main.py --dataset "graph_CN/" \
#         --source_node "${SOURCE_NODES[$i]}" --target_node "${TARGET_NODES[$i]}" \
#         --source_task "${SOURCE_TASKS[$i]}" --target_task "${TARGET_TASKS[$i]}" \
#         --train_batch ${TRAIN_BATCHES[$i]} --test_batch ${TEST_BATCHES[$i]} \
#         --train_batch2 ${TRAIN_BATCHES2[$i]} --test_batch2 ${TEST_BATCHES2[$i]} \
#         --matching_hops ${MATCHING_HOPSS[$i]} --no_matching_loss
#done
#
#
## machine learning
#SOURCE_NODES=("paper" "author" "author" "venue")
#TARGET_NODES=("author" "paper" "venue" "author")
#SOURCE_TASKS=("L2" "L2" "L2" "L2")
#TARGET_TASKS=("L2" "L2" "L2" "L2")
#TRAIN_BATCHES=(400 400 400 35)
#TEST_BATCHES=(50 50 50 15)
#TRAIN_BATCHES2=(10 10 10 10)
#TEST_BATCHES2=(50 50 15 50)
#MATCHING_HOPSS=(1 1 2 2)
#
#length=${#SOURCE_NODES[@]}
#
#for ((i=0;i<=$length;i++))
#do
#    python main.py --dataset "graph_ML/" \
#         --source_node "${SOURCE_NODES[$i]}" --target_node "${TARGET_NODES[$i]}" \
#         --source_task "${SOURCE_TASKS[$i]}" --target_task "${TARGET_TASKS[$i]}" \
#         --train_batch ${TRAIN_BATCHES[$i]} --test_batch ${TEST_BATCHES[$i]} \
#         --train_batch2 ${TRAIN_BATCHES2[$i]} --test_batch2 ${TEST_BATCHES2[$i]} \
#         --matching_hops ${MATCHING_HOPSS[$i]} --no_matching_loss
#done


