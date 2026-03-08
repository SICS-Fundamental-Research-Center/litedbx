.PHONY: clean clean_medical

clean_medical_q1:
	rm -rf .ckpt/medical/Q1*

clean_medical_q3:
	rm -rf .ckpt/medical/Q3*

clean_medical_q8:
	rm -rf .ckpt/medical/Q8*

clean_medical:
	make clean_medical_q1
	make clean_medical_q3
	make clean_medical_q8

clean:
	make clean_medical
