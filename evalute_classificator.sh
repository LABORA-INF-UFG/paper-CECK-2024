#!/bin/bash

function resize_frames {
	echo "+ resize '$1' images to 450x450"
	python3.7 /root/uav_network_analitycs/preprocessamento_imagens.py --dir_images_input '$1/'
	ls "$1/*_450"
        echo " - moving *_450 frames to '$1_450'"
	mkdir "$1_450"
     	mv "$1/*_450" "$1_450"
	echo "  > Done."
}

function evalute_resnet {
        echo "+ evalute resnet to '$1_450' folder"
	python3.7 /root/uav_network_analitycs/evalute_disturb_recog_resnet.py --model_path /root/resnet_image_recog_uav_ResNet29v2_model.072.h5 --base_directory_original '/root/media/$1_450/' --nome_arquivo_resultados '/root/classificador_results/classificador_result_$1.csv'
	echo "  > Done."
}

function run {
	resize_frames $1
	evalute_resnet $1
}

run '/root/media/loss_5'
run '/root/media/loss_10'
run '/root/media/loss_25'
run '/root/media/loss_30'

