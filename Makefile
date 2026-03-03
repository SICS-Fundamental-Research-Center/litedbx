.PHONY: clean clean_medical

clean_medical_q1:
	rm -rf .ckpt/medical/Q1*
	if [ -f .ckpt/medical/state.json ]; then \
		sed -i -E '/^[[:space:]]*"Q1_[^"]*"[[:space:]]*:/d' .ckpt/medical/state.json; \
		sed -i -E ':a;N;$$!ba;s/,\n([[:space:]]*})/\n\1/' .ckpt/medical/state.json; \
	fi

clean_medical_q3:
	rm -rf .ckpt/medical/Q3*
	if [ -f .ckpt/medical/state.json ]; then \
		sed -i -E '/^[[:space:]]*"Q3_[^"]*"[[:space:]]*:/d' .ckpt/medical/state.json; \
		sed -i -E ':a;N;$$!ba;s/,\n([[:space:]]*})/\n\1/' .ckpt/medical/state.json; \
	fi

clean_medical_q8:
	rm -rf .ckpt/medical/Q8*
	if [ -f .ckpt/medical/state.json ]; then \
		sed -i -E '/^[[:space:]]*"Q8_[^"]*"[[:space:]]*:/d' .ckpt/medical/state.json; \
		sed -i -E ':a;N;$$!ba;s/,\n([[:space:]]*})/\n\1/' .ckpt/medical/state.json; \
	fi

clean_medical:
	make clean_medical_q1
	make clean_medical_q3
	make clean_medical_q8

clean:
	make clean_medical
