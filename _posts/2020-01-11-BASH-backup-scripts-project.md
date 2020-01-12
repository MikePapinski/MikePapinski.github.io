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
#!/bin/bash

### Configuration
BACKUP_SOURCE=("/home/mike/Desktop/backup_from/")
BACKUP_DESTINATION="/home/mike/Desktop/backup_to/"
BACKUP_PARENT=20200112-112113-full
BACKUP_FILENAME="archive"

```


# File with library methods:
```bash
#!/bin/bash
#./getDateTime.sh

function MsgType {
	msg="$1"
	error="$2"
	print_now=`date +"%Y-%m-%d %T"`
	if [ "$error" = 1 ]; then
		echo -e "\e[1m${print_now}: \e[31m $msg \e[0m"
		echo -e "\e[1m${print_now}: \e[31m Script stopped. \e[0m"
		exit
	else
		echo -e "\e[1m${print_now}: \e[0;92m $msg \e[0m"
	fi
}

function LoadConfiguration {
	local config="$1"
	if [ -f "$config" ]; then
		MsgType "Configuration file: $config"
	else
		MsgType "Configuration file not found!" 1
	fi
	source "$config"
}

function ValidateBackupSource {
	local source="$1"
	if [ -z "$source" ]; then
		MsgType "No backup source defined!" 1
	fi
	if [ ! -d "$source" ]; then
		MsgType "Backup source not found!" 1
	fi
	MsgType "Backup source	 	: $source "
}

function ValidateBackupDestination {
	local destination="$1"
	if [ -z "$destination" ]; then
		MsgType "No backup destination defined!" 1
	fi
	if [ ! -d "$destination" ]; then
		MsgType "Backup destination not found. Creating ..."
		mkdir -p "$destination"
		if [ ! -d "$destination" ]; then
			MsgType "Could not create the backup folder" 1
		fi
	fi
	MsgType "Backup destination	: $destination "
}

function ValidateBackupParent {
	local parent="$1"
	local destination="$2"
	if [ -z "$parent" ]; then
		MsgType "No backup parent defined!" 1
	fi
	if [ ! -d "$parent" ]; then
		parentDirs=($(find "$destination" -type d -name "$parent"))
		if [ "${#parentDirs[@]}" -gt 1 ]; then
			MsgType "Backup parent is ambivalent!" 1
			for dir in "${parentDirs[@]}"; do
				MsgType "$dir" 1
			done
			MsgType "Cannot continue." 1
		elif [ "${#parentDirs[@]}" -eq 0 ]; then
			MsgType "Backup parent not found!" 1
		fi	
	fi
	MsgType "Backup parent		: $parent "
}

function SetConfigField(){
	local path="$1"
	local field="$2"
	local value="$3"
	sudo sed -i "s/^\($field\s*=\s*\).*\$/\1$value/" $path
}

function FullBackup {
	MsgType "Full Backup requested."
	local source="$1"
	local destination="$2"
	local filename="$3"
	local config="$4"
	local snapshot="level0.snapshot"
	local timestamp=`date +%Y%m%d-%H%M%S`
	local directory="$destination/$timestamp-full"
	local archive="$filename.tar.gz"
	MsgType "Performing full backup..."										
	PerformBackup "$source" "$directory" "$archive" "$snapshot"	
	MsgType "Saving new parent to config..."
	SetConfigField $config BACKUP_PARENT "$timestamp-full"
}

function IncrementalBackup {
	MsgType "Incremental Backup requested."
	local source="$1"
	local destination="$2"
	local filename="$3"
	local config="$4"
	local parent="$5"
	local timestamp=`date +%Y%m%d-%H%M%S`
	local archive="$filename.tar.gz"
	local level=(`find $destination/$parent/ -name "*.snapshot" | wc -l`);
	local directory="$destination/$parent/$timestamp-incremental_$level"
	local lastlevel=$(($level-1))
	local lastfile=(`find $destination/$parent/ -name "level$lastlevel.snapshot"`);
	local parentSnapshot="${lastfile[0]}"
	if [ -z "$parentSnapshot" ]; then
		MsgType "No snapshot found, please do full backup first." 1
	fi
	MsgType "Backup from		: $parentSnapshot"
	MsgType "Level increment		: $lastlevel -> $level"
	local snapshot="level$level.snapshot"				
	MsgType "Snapshot file		: $snapshot"
	MsgType "Creating incremental backup ..."
	mkdir -p "$directory"
	cp -a "$parentSnapshot" "$directory/$snapshot"
	PerformBackup "$source" "$directory" "$archive" "$snapshot"
}

function PerformBackup {
	source="$1"
	destination="$2"
	backupFile="$3"
	snapshotFile="$4"
	mkdir -p "$destination"
	backupStart=$SECONDS	
	local tarOps="-cpvzf"
	tar --listed-incremental="$destination/$snapshotFile" $tarOps "$destination/$backupFile" -C "$source" .
	res=$?
	if [ ! $res -eq 0 ];
	then
		MsgType "Tar failed! ($res)" 1
	else
		backupDuration=$(($SECONDS - $backupStart))
		backupMin=$(($backupDuration / 60))
		backupSec=$(($backupDuration % 60))
		MsgType "Backup completed. Time: ${backupMin}min ${backupSec}sec."
	fi
	return $res
}

function list {
	local destination="$1"
	readDirectory "$destination" 0 "$2"
}

function readDirectory {
	local directory="$1"
	local files=($directory/*)
	local fullDirs=()
	local depth="$2"
	local filename="$3"
	for file in "${files[@]}"; do
		[[ -d "$file" ]] && fullDirs+=("$file")
	done
	for currentDir in "${fullDirs[@]}"; do
		local backupName=`basename "$currentDir"`		
		readBackup "$currentDir" "$filename"
		if [ "${backupInfo[0]}" = false ]; then
			echo -n "[$backupName] "
			MsgType "${backupInfo[1]}" 1
		else
			local archiveDate=`date -r "$currentDir" "+%d.%m.%Y-%H:%M:%S"`		
			local archiveDateNumber=`date -r "$currentDir" "+%s.%N"`
			[ "$depth" -gt 0 ] && type="Level $depth"
			echo "$archiveDate $type Datetime: [$backupName]"
		fi
		readDirectory "$currentDir" "$((depth+1))" "$filename"
	done
}

function readBackup {
	local backupName=`basename $1`
	local filename="$2"
	local files=($(find "$1" -maxdepth 1 -type f -name "$filename.tar.gz" -o -type f -name "$filename.tar.gz2"))
	local archive="${files[0]}"
	files=($1/*.snapshot);
	local snapshot="${files[0]}"
	if [ ! -f "$archive" ]; then
		MsgType "No archive file found!" 1
	elif [ ! -f "$snapshot" ]; then
		MsgType "No snapshot file found!" 1
 	fi

	local level=${snapshot##*level}
	level=${level%.*}
	
	local success=true
	local error=""

	if [ ! -f "$archive" ]; then
		success=false
		error="No archive file found!"
	elif [ ! -f "$snapshot" ]; then
		success=false
		error="No snapshot file found!"
	elif [ -z "$level" ]; then
	 	success=false
	 	error="Could not determine backup level!"
 	fi
	
	if [ "$success" = true ]; then
		backupInfo=(true "$archive" "$snapshot" "$level")
	else
		backupInfo=(false "$error")
	fi

}


function GetCloseSnap {
	MsgType "Browsing backup files..."
	local destination="$2"
	local date_c=$(date -d "$1" +%s)
	local actiontype="$3"
	local filename="$4"
	old_distance="$date_c"
	local old_value=""
	for d in $destination*; do
		for a in $d/*; do
			local date_a=$(date -r "$a" +%s)
			local distance=$(("$date_a"-"$date_c"))
			if (( 0 > $distance )) ;then
				distance=$((distance*-1))			
			fi
			if (( $old_distance > $distance )) ;then
				old_distance="$distance"
				if [[ $a == *"archive.tar.gz"* ]]; then
					old_value=$d
				else
					old_value=$a
				fi
			fi
		done
	done
	MsgType "Closest date is: $(date -r "$old_value" +%d.%m.%Y-%H:%M:%S)"
	MsgType "Backup found: $old_value"
	if (( $actiontype == 1)) ; then
		MsgType "Restoring folder..."
		local lastfile=(`find $old_value/ -name "level*.snapshot"`);
		local parentSnapshot="${lastfile[0]}"
		echo $parentSnapshot
		restore "$old_value" "$filename"
		MsgType "Folder restored."
	else
		MsgType "Providing metadata..."
		tar tzf "$old_value/archive.tar.gz"
		MsgType "Metadata provided."
	fi
}


function restore {
	local backuppath="$1"
	local filename="$2"
	MsgType "Restoration requested." 
	readBackup "$backuppath" "$filename"
	if [ "${backupInfo[0]}" = false ]; then
		MsgType "${backupInfo[1]}" 1
	fi
	local archive="${backupInfo[1]}"
	local snapshot="${backupInfo[2]}"
	local level="${backupInfo[3]}"
	echo "Archive file            : `basename "$archive"`"
	echo "Snapshot file           : `basename "$snapshot"`"
	echo "Level                   : $level"
	MsgType "Building incremental backup chain ..."
	local backupChain=($archive)
	local currentDirectory=`dirname "$archive"`
	local currentLevel="$level"

	local root_folder="${backuppath%/*}"
	while [ "$currentLevel" -gt 0 ]; do

		local findsnapshot=(`find $root_folder/ -name "*level$((currentLevel-1)).snapshot"`);
		local snapshotfound="${findsnapshot[0]}"
		local validpath="${snapshotfound%/*}"
		MsgType "Restoring from: $validpath"
		readBackup "$validpath" "$filename"
		if [ "${backupInfo[0]}" = false ]; then
			MsgType "${backupInfo[1]}" 1
		fi
		backupChain+=(${backupInfo[1]})
		((currentLevel--))
	done
	printf '%s\n' "${backupChain[@]}"
	echo "Restoring backup ... "
	local chainLastIndex=$((${#backupChain[@]}-1))
	for ((chainIndex=$chainLastIndex; chainIndex >= 0; chainIndex--)); do
		local backupArchive="${backupChain[$chainIndex]}"
		local backupDir=`dirname "$backupArchive"`
		local backupName=`basename "$backupDir"`
		local tarOps="-x"
		tarOps="${tarOps}v"
		tarOps="${tarOps}z"		
		tarOps="${tarOps}f"
		MsgType "[$backupName] ... "
		tar $tarOps "$backupArchive" -C "$BACKUP_SOURCE"		
		res=$?
		if [[ "$res" -eq 0 ]]; then
			 MsgType "Success"
		else	
			echo "error $res"
			MsgType "Could not restore backup!" 1
		fi			
	done
}



```



# Main file to run
```bash
#!/bin/bash
. lib.sh
#################################################################
# Methods imported from lib:
# MsgType()
# LoadConfiguration(config)
# ValidateBackupSource(source)
# ValidateBackupDestination(destination)
# ValidateBackupParent(parent,destination)
# SetConfigField(path,field,value)
# FullBackup(source,destination,filename,config)
# IncrementalBackup(source,destination,filename,snapshot,parent)
# PerformBackup(source,destination,backupfile,snapshotfile)
#################################################################
MsgType "Welcome to incremental backups tool."

read -r -d '' welcome_print << EOM
Usage:
 *backup-full: Perform full backup of the folder.
 *backup-increment: Perform incremental backup.
 *restore: Restore folder to state from given time [date]
	-example:  restore '2020/01/11 09:35:00'
 *show: show files in backup files  [date]
	-example:  show '2020/01/11 09:35:00'
 *list: list all backup files

EOM
echo  "$welcome_print"

MYDIR="$(dirname "$(readlink -f "$0")")"
CONFIG_FILE="$MYDIR/configuration.conf"
LoadConfiguration $CONFIG_FILE
##################################################
# Fields loaded from the configuration.conf file:
# BACKUP_SOURCE
# BACKUP_DESTINATION
# BACKUP_PARENT
# BACKUP_FILENAME
##################################################

ValidateBackupSource $BACKUP_SOURCE
ValidateBackupDestination $BACKUP_DESTINATION

ACTION="$1"
ARGUMENT="$2"
case "$ACTION" in
	backup-full)
		FullBackup "$BACKUP_SOURCE" "$BACKUP_DESTINATION" "$BACKUP_FILENAME" "$CONFIG_FILE";;
	backup-increment)
		ValidateBackupParent "$BACKUP_PARENT" "$BACKUP_DESTINATION"
		IncrementalBackup "$BACKUP_SOURCE" "$BACKUP_DESTINATION" "$BACKUP_FILENAME" "$CONFIG_FILE" "$BACKUP_PARENT";;
	list)
		list "$BACKUP_DESTINATION" "$BACKUP_FILENAME";;
	restore)
		GetCloseSnap "$ARGUMENT" "$BACKUP_DESTINATION" 1 "$BACKUP_FILENAME";;
	show)
		GetCloseSnap "$ARGUMENT" "$BACKUP_DESTINATION" 0;;
	*)
		[ -z "$ACTION" ] && MsgType "No action supplied!" 1
		MsgType "Unknown action! ($ACTION)" 1
esac
		
exit 0

```


# How to run this backup script??
### Example for full backup:
```bash
$ sudo ./backup.sh backup-full
```


# How to schedule this as reccurent task???
#Use Cron jobs on linux!

### How Do I install or create or edit my own cron jobs?
To edit or create your own crontab file, type the following command at the UNIX / Linux shell prompt:

```bash
$ crontab -e
```

Syntax of crontab (field description)
The syntax is:
```bash
1 2 3 4 5 /path/to/command arg1 arg2
```
OR

```bash
1 2 3 4 5 /root/backup.sh
```
Where,

```bash
1: Minute (0-59)
2: Hours (0-23)
3: Day (0-31)
4: Month (0-12 [12 == December])
5: Day of the week(0-7 [7 or 0 == sunday])
/path/to/command – Script or command name to schedule
Easy to remember format:

* * * * * command to be executed
- - - - -
| | | | |
| | | | ----- Day of week (0 - 7) (Sunday=0 or 7)
| | | ------- Month (1 - 12)
| | --------- Day of month (1 - 31)
| ----------- Hour (0 - 23)
------------- Minute (0 - 59)
```

Your cron job looks as follows for system jobs:


```bash
1 2 3 4 5 USERNAME /path/to/command arg1 arg2
```

OR

```bash
1 2 3 4 5 USERNAME /path/to/script.sh
```


# Example: Run backup cron job script
If you wished to have a script named /root/backup.sh run every day at 3am, your crontab entry would look like as follows. First, install your cronjob by running the following command:

```bash
# crontab -e
```

Append the following entry:

```bash
0 3 * * * /root/backup.sh backup-full
0 3 * * * /root/backup.sh backup-incremental
```

Save and close the file.