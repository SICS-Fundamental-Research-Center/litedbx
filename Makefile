.PHONY: clean clean_medical

clean_medical:
	rm -rf .ckpt/medical/*

clean:
	make clean_medical
