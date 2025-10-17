

for file in ["https://drive.google.com/file/d/1bJ0C2OvVdBjyyL-KlaCPdqchFB_mvdA7/view?usp=drive_link", "https://drive.google.com/file/d/1yKdvBYFgK6AzHdh66Nuem5TRrg0Uy16a/view?usp=drive_link", "https://drive.google.com/file/d/17ISM62qq2ohpgBBm5f4DaGioIKQSozFJ/view?usp=drive_link"]:
  !gdown --fuzzy $file
!mv pretr* models/.
