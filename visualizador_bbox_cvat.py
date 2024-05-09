import csv
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import  utils
import numpy as np
import  os
import tensorflow as tf
import statistics


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-cba ' ,'--caminho_base_arquivos' ,type=str, default ='' ,help='')
	parser.add_argument('-cdbb ', '--caminho_arquivo_cvat', type=str, default='', help='')
	parser.add_argument('-dsa ', '--diretorio_saida_frames_bbox', type=str, default='', help='')
	args = parser.parse_args()
	caminho_base_arquivos = args.caminho_base_arquivos
	caminho_arquivo_cvat = args.caminho_arquivo_cvat
	diretorio_saida_frames_bbox = args.diretorio_saida_frames_bbox
	if not os.path.exists(diretorio_saida_frames_bbox):
		os.makedirs(diretorio_saida_frames_bbox)
	dict_bbox_gt = utils.carregar_cvat_ground_truth(caminho_arquivo_cvat)
	bbox_red_color = (255, 0, 0)
	for path, currentDirectory, files in os.walk(caminho_base_arquivos):
		for img_path in files:
			if not img_path.endswith('.jpg'):
				continue
			absolute_image_file_path = os.path.join(path, img_path)
			print(absolute_image_file_path)
			original_image = cv2.imread(absolute_image_file_path)
			original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
			nome_arquivo_frame = img_path.split('/')[-1]
			ground_truth_frame = dict_bbox_gt[nome_arquivo_frame]
			for index_gt, gt_bbox in enumerate(ground_truth_frame):
				x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
				cv2.rectangle(original_image, (int(x_topleft_gt), int(y_topleft_gt)), (int(x_bottomright_gt), int(y_bottomright_gt)), bbox_red_color, 1)
			image = Image.fromarray(original_image)
			image.save(os.path.abspath(os.path.join(diretorio_saida_frames_bbox, img_path)))


if __name__ == '__main__':
	main()