.PHONY: clean clean_medical

clean_ecomm_q1:
	rm -rf .ckpt/ecomm/Q1*


clean_mmqa_q3a:
	rm -rf .ckpt/mmqa/Q3a*

clean_mmqa_q3f:
	rm -rf .ckpt/mmqa/Q3f*

clean_mmqa_q6a:
	rm -rf .ckpt/mmqa/Q6a*

clean_mmqa_q6b:
	rm -rf .ckpt/mmqa/Q6b*

clean_mmqa_q6c:
	rm -rf .ckpt/mmqa/Q6c*

clean_medical_q1:
	rm -rf .ckpt/medical/Q1*

clean_medical_q3:
	rm -rf .ckpt/medical/Q3*

clean_medical_q8:
	rm -rf .ckpt/medical/Q8*

clean_ecomm:
	make clean_ecomm_q1

clean_mmqa:
	make clean_mmqa_q3a
	make clean_mmqa_q3f
	make clean_mmqa_q6a
	make clean_mmqa_q6b
	make clean_mmqa_q6c

clean_medical:
	make clean_medical_q1
	make clean_medical_q3
	make clean_medical_q8

clean:
	make clean_ecomm
	make clean_mmqa
	make clean_medical
