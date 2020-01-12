---
layout: post
title: Bash incremental backup scripts
author: MikePapinski
summary: This is a simple bash backup project that i had to do fo the university.
categories: [Bash]
image: assets/images/posts/8/post_8.JPG
---




# What is the idea?
We are going to create 3 files.
    -configuration.config <-- this is the file to store settings
    -backup.sh <-- main file of running the backup
    -lib.sh <-- a small library with all backup methods listed.

## Functions required:
 *backup-full: Perform full backup of the folder.
 *backup-increment: Perform incremental backup.
 *restore: Restore folder to state from given time [date]
	-example:  restore '2020/01/11 09:35:00'
 *show: show files in backup files  [date]
	-example:  show '2020/01/11 09:35:00'
 *list: list all backup files



# File with configuration:
```bash
%matplotlib inline
%load_ext autoreload
%autoreload 2
%matplotlib notebook
```