#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define FREE_SIZE 151
#define I_SIZE 200
#define BLOCK_SIZE 1024
#define ADDR_SIZE 11
#define INPUT_SIZE 256
#define INODE_SIZE 64

// Superblock Structure, size of superblock has been expanded to 1024 bytes
typedef struct {
	unsigned short nfree;  // number of free blocks in the following 100 bloacks
	unsigned short fmod; //used as a flag to indicate that the super-block has changed and should be copied to the disk during the next periodic update of file system information
	unsigned short ninode;
	char flock; //flags maintained in the core fopy of the file system while it is mounted ahd its values on disk is immaterial
	char ilock; //flags maintained in the core fopy of the file system while it is mounted ahd its values on disk is immaterial
	unsigned int isize;           //the number of blocks devoted to the i-list
	unsigned int free[151];
	unsigned short inode[200];
	unsigned int fsize; //the first block not potentially available for allocation to a file.
	unsigned short time[2]; //the last time the super-block of the file system was changed
} super_block;

super_block superBlock;

// I-Node Structure, I-Node size has been expanded to 64 bytes
typedef struct {
	unsigned short flags; // determines all properties of the file/directory. eg. small or large file
	unsigned short nlinks;     //number of links to file
	unsigned short uid;        //user ID of owner
	unsigned short gid;        //group ID of owner
	unsigned int size;
	unsigned int addr[ADDR_SIZE];   //block number of device number
	unsigned short actime[2];       //time of last access
	unsigned short modtime[2];      //time of last modification
} inode_type;

inode_type inode;

//Directory Structure, 16 bytes
typedef struct {
	unsigned short inode;
	char filename[14];
} dir_type;

int fileDescriptor;		//file descriptor
const unsigned short inode_alloc_flag = 0100000;
const unsigned short dir_flag = 040000;
const unsigned short large_file = 010000;
const unsigned short plain_file = 000000;
const unsigned short dir_access_rights = 000777; // User, Group, & World have all access privileges

unsigned int num_Of_Blocks = 0, num_Of_Inodes = 0;
unsigned char empty_Buffer[BLOCK_SIZE] = { 0 };
char *fileSystemName;
char *currentPath;

void commandControl();
int preInitialization();
int initfs(char* path, unsigned short total_blocks, unsigned short total_inodes);
void updateSuperBlock();
void createRootDir();
void freeBlocks(unsigned int frist_free_block_num,
		unsigned int last_free_block_num);
void add_block_to_free_list(unsigned int blockNum);
unsigned int allocate_block_to_file();
int get_inode_number_by_path(char* path);
int get_inode_number_by_parent(char* filename, int inodeNum);
void *getFileName_and_parentPath(char filePath[], const char *delim,
		char **fileName, char **parentPath);
unsigned short allocate_inode();
int find_dir_entry_position_by_parent_inode(int parent_inode);
void cpin(char externalFile[], char v6File[]);
void cpout(char* internal_path, char* external_path);
void mkdirectory(char *path);
void ls();
void openExitingFileSystem();
void rmFile(char* path);
void rmDir(char *path);
void remove_dir_entry_in_parent(int parent_inode_num, char *Name);
void pwd();
void cd(char *path);

int main() {
	commandControl();
	return 0;
}

void commandControl() {
	printf("Size of super block = %ld , size of i-node = %ld\n",
			sizeof(superBlock), sizeof(inode));
	char *splitter, cmd[256];
	printf("\nEnter command\n");
	while (1) {
		if(currentPath != NULL) printf("%s --> ", currentPath);
		scanf(" %[^\n]s", cmd);
		splitter = strtok(cmd, " ");

		if (strcmp(splitter, "initfs") == 0) {
			preInitialization();
			splitter = NULL;
		} else if (strcmp(splitter, "cpin") == 0) {
			char *externalFile;
			char *v6File;
			externalFile = strtok(NULL, " ");
			v6File = strtok(NULL, " ");
			if (strstr(v6File, "/") == NULL ) {
				//support relative path
				char temp[1000];
				strcpy(temp, currentPath);
				strcat(temp, "/");
				strcat(temp, v6File);
				cpin(externalFile, temp);
			} else {
				cpin(externalFile, v6File);
			}
			printf("\n");
		} else if (strcmp(splitter, "cpout") == 0) {
			char *externalFile;
			char *internalFile;
			internalFile = strtok(NULL, " ");
			externalFile = strtok(NULL, " ");
			if (strstr(internalFile, "/") == NULL ) {
				//support relative path
				char temp[1000];
				strcpy(temp, currentPath);
				strcat(temp, "/");
				strcat(temp, internalFile);
				cpout(temp, externalFile);
			} else {
				cpout(internalFile, externalFile);
			}
			printf("\n");
		} else if (strcmp(splitter, "mkdir") == 0) {
			char * dir;
			dir = strtok(NULL, " ");
			if (strstr(dir, "/") == NULL ) {
				//support relative path
				char temp[1000];
				strcpy(temp, currentPath);
				strcat(temp, "/");
				strcat(temp, dir);
				mkdirectory(temp);
			} else {
				mkdirectory(dir);
			}
			printf("\n");
		} else if (strcmp(splitter, "rm") == 0) {
			char *path;
			path = strtok(NULL, " ");
			if (strstr(path, "/") == NULL ) {
				//support relative path
				char temp[1000];
				strcpy(temp, currentPath);
				strcat(temp, "/");
				strcat(temp, path);
				rmFile(temp);
			} else {
				rmFile(path);
			}
			printf("\n");
		} else if (strcmp(splitter, "ls") == 0) {
			ls();
			printf("\n");
		} else if (strcmp(splitter, "pwd") == 0) {
			pwd();
			printf("\n");
		} else if (strcmp(splitter, "cd") == 0) {
			char *path;
			path = strtok(NULL, " ");
			if (strstr(path, "/") == NULL ) {
				//support relative path
				if (strstr(path, "..")) {
					//handle with ..
					cd(path);
				} else {
					char temp1[1000];
					strcpy(temp1, currentPath);
					strcat(temp1, "/");
					strcat(temp1, path);
					cd(temp1);

				}
			} else {
				cd(path);
			}
			printf("\n");
		} else if (strcmp(splitter, "rmdir") == 0) {
			char * dir;
			dir = strtok(NULL, " ");
			if (strstr(dir, "/") == NULL ) {
				//support relative path
				char temp[1000];
				strcpy(temp, currentPath);
				strcat(temp, "/");
				strcat(temp, dir);
				rmDir(temp);
			} else {
				rmDir(dir);
			}

			printf("\n");

		} else if (strcmp(splitter, "open") == 0) {
			char * path;
			path = strtok(NULL, " ");
			if(open(path,O_RDONLY) == -1){
				printf("Cannot open file !\n");
			}else{
				printf("Successfully open file !\n");
			}

		} else if (strcmp(splitter, "openfs") == 0) {
			openExitingFileSystem();
			printf("\n");
		} else if (strcmp(splitter, "q") == 0) {
			//update superblock and quit fileSystem
			lseek(fileDescriptor, BLOCK_SIZE, 0);
			write(fileDescriptor, &superBlock, BLOCK_SIZE);
			close(fileDescriptor);
			printf("\nEXITING FILE SYSTEM NOW....\n");
			break;
		} else {
			printf("\nInvalid command\n ");
			printf("\n");
		}
	}

}

//This function will first validate input filepath.
//If it is valid then it will call function initfs to init the file System
int preInitialization() {
	char *n1, *n2;
	char temp[1000];
	char *filePath = strtok(NULL, " ");
	strcpy(temp, filePath);
	n1 = strtok(NULL, " ");
	n2 = strtok(NULL, " ");

	char msg[1000][20];
	int i = 0;
	char *result = strtok(temp, "/");
//get fileSystemName
	for (i = 0; result != NULL ; i++) {
		strcpy(msg[i], result);
		result = strtok(NULL, "/");
	}
	fileSystemName = msg[i - 1];
	currentPath = msg[i - 1];
	if (access(filePath, F_OK) != -1) {
		if ((fileDescriptor = open(filePath, O_RDWR, 0600)) == -1) {
			printf(
					"\n filesystem already exists but open() failed with error [%s]\n",
					strerror(errno));
			return 1;
		}
		printf("filesystem already exists and the same will be used.\n");

	} else {

		if (!n1 || !n2)
			printf(
					" All arguments(path, number of inodes and total number of blocks) have not been entered\n");
		else {
			//initializing global variables
			num_Of_Blocks = atoi(n1);
			num_Of_Inodes = atoi(n2);

			if (initfs(filePath, num_Of_Blocks, num_Of_Inodes)) {
				printf("The file system is initialized\n");
				//printf current path
			} else {
				printf("Error initializing file system. Exiting... \n");
				return 1;
			}
		}
	}
	return 0;
}

//This function is used to initialize fileSystem based on parameters which are entered by user
int initfs(char* path, unsigned short total_blocks, unsigned short total_inodes) {
	unsigned short i = 0;

	if ((fileDescriptor = open(path, O_RDWR | O_CREAT, 0700)) == -1) {
		printf("\n open() failed with the following error [%s]\n",
				strerror(errno));
		return 0;
	}

	if (num_Of_Blocks == 0 || num_Of_Inodes == 0
			|| num_Of_Inodes > num_Of_Blocks * 16) {
		printf("incorrect block number or inodes number\n");
		return 0;
	}

//initializing superblock
	superBlock.nfree = 0;
	superBlock.fmod = 0;
	superBlock.ninode = I_SIZE;
	superBlock.flock = 0;
	superBlock.ilock = 0;
	superBlock.isize = num_Of_Inodes / 16;
	superBlock.isize += (superBlock.isize * 16) >= num_Of_Inodes ? 0 : 1;
	superBlock.fsize = num_Of_Blocks;
	superBlock.time[0] = (unsigned short) time(NULL );
	superBlock.time[1] = 1970;

//initializing free array to 0 to remove junk data. free array will be stored with data block numbers shortly.
	for (i = 0; i < FREE_SIZE; i++)
		superBlock.free[i] = 0;

//initializing the inode array to inode numbers
//inode 1 is allocated to root and i-number begins at 1
	for (i = 0; i < I_SIZE; i++)
		superBlock.inode[i] = i + 2;

//update superblock
	updateSuperBlock();

// writing zeroes to all blocks devoted to i-list
	for (i = 0; i < superBlock.isize; i++)
		write(fileDescriptor, &empty_Buffer, BLOCK_SIZE);

//create root directory
	createRootDir();

//add all free blocks into free array
	freeBlocks(2 + superBlock.isize + 1, num_Of_Blocks - 1);

	return 1;
}

void updateSuperBlock() {
	superBlock.time[0] = (unsigned int) time(NULL );
	lseek(fileDescriptor, BLOCK_SIZE, SEEK_SET);
	write(fileDescriptor, &superBlock, BLOCK_SIZE);
}

void createRootDir() {
	dir_type root1, root2;
	int root_data_block = 2 + superBlock.isize; // Allocating first data block to root directory
	int i;

	root1.inode = 1;   // root directory's inode number is 1.
	strcpy(root1.filename, fileSystemName);

	inode.flags = inode_alloc_flag | dir_flag | large_file | dir_access_rights; // flag for root directory
	inode.nlinks = 0;
	inode.uid = 0;
	inode.gid = 0;
	inode.size = 32;
	inode.addr[0] = root_data_block;

	for (i = 1; i < ADDR_SIZE; i++) {
		inode.addr[i] = 0;
	}

	inode.actime[0] = (unsigned short) time(NULL );
	inode.actime[1] = 1970;
	inode.modtime[0] = (unsigned short) time(NULL );
	inode.modtime[1] = 1970;

	lseek(fileDescriptor, 2 * BLOCK_SIZE, 0);
	write(fileDescriptor, &inode, INODE_SIZE);

	lseek(fileDescriptor, root_data_block * BLOCK_SIZE, 0);
	write(fileDescriptor, &root1, 16);

	root2.filename[0] = '.';
	root2.filename[1] = '.';
	root2.filename[2] = '\0';
	root2.inode = 1;
	lseek(fileDescriptor, root_data_block * BLOCK_SIZE + 16, 0);
	write(fileDescriptor, &root2, 16);

}

//add all free blocks into free array(only used during initialization phase of file system)
void freeBlocks(unsigned int frist_free_block_num,
		unsigned int last_free_block_num) {
//denotes no free block left
	superBlock.free[0] = 0;
	superBlock.nfree++;

	updateSuperBlock();

	unsigned int pointer;

	for (pointer = frist_free_block_num; pointer <= last_free_block_num;
			pointer++) {
		add_block_to_free_list(pointer);
	}
}

void add_block_to_free_list(unsigned int blockNum) {
	if (blockNum >= superBlock.fsize) {
		printf("Invalid block number ! Block cannot be allocated to a file!\n");
		return;
	}

	if (superBlock.nfree == FREE_SIZE) {
		//write nfree and free array at the beginning of this block
		//set free[0] equals number of this block
		lseek(fileDescriptor, blockNum * BLOCK_SIZE, SEEK_SET);
		write(fileDescriptor, &superBlock.nfree, 2);

		lseek(fileDescriptor, blockNum * BLOCK_SIZE + 2, SEEK_SET);
		write(fileDescriptor, &superBlock.free, 4 * superBlock.nfree);

		superBlock.nfree = 0;
	} else {
		// writing 0 to this block to get rid of junk data
		lseek(fileDescriptor, blockNum * BLOCK_SIZE, 0);
		write(fileDescriptor, &empty_Buffer, BLOCK_SIZE);
	}
//set free[superBlock.nfree] to the freed block's number and increment nfree
	superBlock.free[superBlock.nfree] = blockNum;
	superBlock.nfree++;

	updateSuperBlock();
}

unsigned int allocate_block_to_file() {
	superBlock.nfree--;

	unsigned int blockNum;

	if (superBlock.nfree == 0) {
		blockNum = superBlock.free[0];

		lseek(fileDescriptor, blockNum * BLOCK_SIZE, SEEK_SET);
		read(fileDescriptor, &superBlock.nfree, 2);

		lseek(fileDescriptor, blockNum * BLOCK_SIZE + 2, SEEK_SET);
		read(fileDescriptor, &superBlock.free, 4 * superBlock.nfree);
	} else {
		blockNum = superBlock.free[superBlock.nfree];
	}

	updateSuperBlock();

	lseek(fileDescriptor, blockNum * BLOCK_SIZE, SEEK_SET);
	write(fileDescriptor, &empty_Buffer, BLOCK_SIZE);
	return blockNum;
}

int get_inode_number_by_path(char* path) {
	const char s[2] = "/";
	char *token;
	token = strtok(path, s);
	int i = 1;
	if (token == NULL )
		return -1;
	while (token != NULL ) {
		if (get_inode_number_by_parent(token, i) == -1)
			return -1;

		i = get_inode_number_by_parent(token, i);
		token = strtok(NULL, s);
	}
	return i;
}

//get addr[0] in inode which denotes inode number of the block contains directory information
//and then check whether filename is in block(addr[0]).
int get_inode_number_by_parent(char *filename, int inodeNum) {
	dir_type temp_type;
	int addr0, i;
	lseek(fileDescriptor, 2 * BLOCK_SIZE + (inodeNum - 1) * INODE_SIZE + 12,
			SEEK_SET);
//addr0 denotes block number of the block contains directory information
	read(fileDescriptor, &addr0, 4);
	for (i = 0; i < 64; i++) {
		lseek(fileDescriptor, addr0 * BLOCK_SIZE + i * 16, SEEK_SET);
		read(fileDescriptor, &temp_type, 16);
		if (strcmp(temp_type.filename, filename) == 0) {
			return temp_type.inode;
		}
	}
	return -1;
}

void *getFileName_and_parentPath(char filePath[], const char *delim,
		char **fileName, char **parentPath) {
	char *result;
	result = strtok(filePath, delim);
	char msg[1000][20];
	char pPath[1000] = { 0 };
	int i = 0;
	int j = 0;
//get fileName
	for (i = 0; result != NULL ; i++) {
		strcpy(msg[i], result);
		result = strtok(NULL, delim);
	}
	*fileName = msg[i - 1];
//get parentPath
	for (j = 0; j < i - 1; j++) {
		char d[2] = "/";
		if (j != 0) {
			strcat(pPath, d);
		}
		strcat(pPath, msg[j]);
	}
	*parentPath = pPath;
	return 0;
}

//return inode number which has been allocated to this file
unsigned short allocate_inode() {
	unsigned short inodeNum;

	 if (superBlock.ninode == 0) {
	  unsigned int i = 0;
	  inode_type freeinode;
	  while (i < num_Of_Inodes) {
	   lseek(fileDescriptor, 2 * BLOCK_SIZE + i * INODE_SIZE, SEEK_SET);
	   read(fileDescriptor, &freeinode, INODE_SIZE);
	   if ((freeinode.flags & 0100000) == 0) {
	    inodeNum = i;
	    break;
	   }
	   i++;
	  }
	  while (i < num_Of_Inodes && superBlock.ninode < I_SIZE) {
	   lseek(fileDescriptor, 2 * BLOCK_SIZE + i * INODE_SIZE, SEEK_SET);
	   read(fileDescriptor, &freeinode, INODE_SIZE);
	   if ((freeinode.flags & 0100000) == 0) {
	    lseek(fileDescriptor,
	      1 * BLOCK_SIZE + 2 + 2 + 2 + 2 + 4 * FREE_SIZE
	        + superBlock.ninode * 2, SEEK_SET);
	    write(fileDescriptor, &i, 2);
	                superBlock.ninode++;
	   }
	   i++;
	  }
	 } else {
	        superBlock.ninode--;
	  inodeNum = superBlock.inode[superBlock.ninode];
	 }

	 updateSuperBlock();

	 return inodeNum;

}

int find_dir_entry_position_by_parent_inode(int parent_inode) {
	int position;
	int addr0;
	int i;
	dir_type dir_entry;

	lseek(fileDescriptor, 2 * BLOCK_SIZE + (parent_inode - 1) * INODE_SIZE + 12,
			SEEK_SET);
	read(fileDescriptor, &addr0, 4);

//find the first available position in parent dir datablock
	for (i = 0; i < 64; i++) {
		lseek(fileDescriptor, addr0 * BLOCK_SIZE + i * 16, SEEK_SET);
		read(fileDescriptor, &dir_entry, 16);

		if (dir_entry.inode == 0) {
			position = addr0 * BLOCK_SIZE + 16 * i;
			return position;
		}
	}
	return 0;
}

void cpin(char externalFile[], char v6File[]) {
	char *fileName = NULL;
	char *parentPath = NULL;
	const char delim[2] = "/";
//validate v6File
	getFileName_and_parentPath(v6File, delim, &fileName, &parentPath);
	int parentInode = 0;
	parentInode = get_inode_number_by_path(parentPath);
	if (parentInode == -1) {
		printf("Invalid path !\n");
		return;
	}

	if (get_inode_number_by_parent(fileName, parentInode) != -1) {
		printf("File exists !\n");
		return;
	}

	unsigned short newFileInodeNum = allocate_inode();

	dir_type newdir;
	newdir.inode = newFileInodeNum;
	strcpy(newdir.filename, fileName);
//write into parent dir entry
	lseek(fileDescriptor, find_dir_entry_position_by_parent_inode(parentInode),
			SEEK_SET);
	write(fileDescriptor, &newdir, 16);
//inode for this v6file
	inode_type temp_inode;
	int index;
	for (index = 0; index <= ADDR_SIZE - 1; index++) {
		temp_inode.addr[index] = 0;
	}
//copy external file contents into v6file
	int exFilefileDescriptor = open(externalFile, O_RDWR);
	if (exFilefileDescriptor == -1) {
		printf("Invalid external file !");
		return;
	}
	int fileSize = lseek(exFilefileDescriptor, 0, SEEK_END);
	int totalNeededBlock = fileSize / 1024 + 1;
	temp_inode.size = fileSize;
	printf("fileSize:%d \n", fileSize);

	int indexOfAddr = 0;
	int firstIndex = 0, secondIndex = 0, thirdIndex = 0;
	int firstBlock = 0, secondBlock = 0, thirdBlock = 0;
	int single_indirect_index = 0;
	int single_indirect_block = 0;
	char buffer[BLOCK_SIZE] = { 0 };

//the following lseek is used to locate external file
	int i = 0;
	lseek(exFilefileDescriptor, i * BLOCK_SIZE, SEEK_SET);

	while (read(exFilefileDescriptor, &buffer, BLOCK_SIZE) != 0) {
		int currentBlk = allocate_block_to_file();

		//top 10 block stored in address array are single indirect block
		if (indexOfAddr < ADDR_SIZE - 1) {

			if (single_indirect_index == 0) {
				single_indirect_block = allocate_block_to_file();
				lseek(fileDescriptor,
						2 * BLOCK_SIZE + (newFileInodeNum - 1) * 64 + 12
								+ indexOfAddr * 4, SEEK_SET);
				write(fileDescriptor, &single_indirect_block, 4);
				temp_inode.addr[indexOfAddr] = single_indirect_block;
			}

			lseek(fileDescriptor,
					single_indirect_block * BLOCK_SIZE
							+ single_indirect_index * 4, SEEK_SET);
			write(fileDescriptor, &currentBlk, 4);

			if (single_indirect_index < 255) {
				single_indirect_index++;
			} else {
				single_indirect_index = 0;
				indexOfAddr++;
			}
		} else {
			//the last block in address array is triple indirect block
			if (firstIndex == 0 && secondIndex == 0 && thirdIndex == 0) {
				firstBlock = allocate_block_to_file();
				secondBlock = allocate_block_to_file();
				thirdBlock = allocate_block_to_file();
				lseek(fileDescriptor,
						2 * BLOCK_SIZE + (newFileInodeNum - 1) * 64 + 12
								+ (ADDR_SIZE - 1) * 4, SEEK_SET);
				write(fileDescriptor, &firstBlock, 4);
			}

			lseek(fileDescriptor, firstBlock * BLOCK_SIZE + firstIndex * 4,
					SEEK_SET);
			write(fileDescriptor, &secondBlock, 4);
			lseek(fileDescriptor, secondBlock * BLOCK_SIZE + secondIndex * 4,
					SEEK_SET);
			write(fileDescriptor, &thirdBlock, 4);
			lseek(fileDescriptor, thirdBlock * BLOCK_SIZE + thirdIndex * 4,
					SEEK_SET);
			write(fileDescriptor, &currentBlk, 4);

			if (thirdIndex < 255) {
				thirdIndex++;
				thirdBlock = allocate_block_to_file();
			} else {
				thirdIndex = 0;
				thirdBlock = allocate_block_to_file();
				secondIndex++;
			}

			if (secondIndex == 256) {
				secondIndex = 0;
				secondBlock = allocate_block_to_file();
				firstIndex++;
			}

			if (firstIndex == 256) {
				printf("file size is larger than max size\n");
			}
			temp_inode.addr[indexOfAddr] = firstBlock;
		}

		if (totalNeededBlock == 1) {
			lseek(fileDescriptor, currentBlk * BLOCK_SIZE, SEEK_SET);
			write(fileDescriptor, buffer, (fileSize % 1024));
			break;
		} else {
			lseek(fileDescriptor, currentBlk * BLOCK_SIZE, SEEK_SET);
			write(fileDescriptor, buffer, sizeof(buffer));
			totalNeededBlock--;
		}

		i++;
		lseek(exFilefileDescriptor, i * BLOCK_SIZE, SEEK_SET);

	}
//printf( "%s is copied into  %s successfully.\n",externalFile,v6File);
//initial inode
	if (temp_inode.addr[ADDR_SIZE - 1] != 0) {
		temp_inode.flags = inode_alloc_flag | plain_file | large_file
				| dir_access_rights;
		// flag for large plain file
	} else {
		temp_inode.flags = inode_alloc_flag | plain_file | dir_access_rights; // flag for small plain file
	}

	temp_inode.nlinks = 0;
	temp_inode.uid = 0;
	temp_inode.gid = 0;
	temp_inode.actime[0] = (unsigned short) time(NULL );
	temp_inode.actime[1] = 1970;
	temp_inode.modtime[0] = (unsigned short) time(NULL );
	temp_inode.modtime[1] = 1970;

	lseek(fileDescriptor, 2 * BLOCK_SIZE + (newFileInodeNum - 1) * INODE_SIZE,
			0);
	write(fileDescriptor, &temp_inode, INODE_SIZE);

	return;
}

void cpout(char* internal_path, char* external_path) {
//printf("%s", internal_path);
//printf("%s", external_path);
	int external_fd = open(external_path, O_RDWR | O_TRUNC | O_CREAT, 0777);
	int inode_no = get_inode_number_by_path(internal_path);
	int count = 0;

	if (external_fd == -1) {
		printf("Invalid external file !");
		return;
	}
	if (inode_no == -1) {
		printf("Invalid internal file !");
		return;
	}

	inode_type temp_inode;
	int fileSize = temp_inode.size;
//get information from inode

	lseek(fileDescriptor, 2 * BLOCK_SIZE + (inode_no - 1) * INODE_SIZE,
			SEEK_SET);
	read(fileDescriptor, &temp_inode, INODE_SIZE);
//test
	lseek(fileDescriptor, 2 * BLOCK_SIZE + (inode_no - 1) * INODE_SIZE + 8,
			SEEK_SET);
	read(fileDescriptor, &fileSize, 4);
//
//int fileSize = temp_inode.size;
	int totalNeededBlock = fileSize / 1024 + 1;
	printf("fileSize:%d \n", fileSize);
	char buffer[1024];

	int index = 0;
	for (index = 0; index < ADDR_SIZE; index++) {
		if (temp_inode.addr[index] == 0) {
			break;
		}
		int data_block_num = 0;
		if (index < ADDR_SIZE - 1) {
			int i;
			for (i = 0; i < 256; i++) {
				lseek(fileDescriptor,
						temp_inode.addr[index] * BLOCK_SIZE + i * 4, SEEK_SET);
				read(fileDescriptor, &data_block_num, 4);
				if (data_block_num == 0)
					break;
				lseek(fileDescriptor, data_block_num * BLOCK_SIZE, SEEK_SET);
				read(fileDescriptor, &buffer, BLOCK_SIZE);

				if (totalNeededBlock == 1) {
					lseek(external_fd, count * BLOCK_SIZE, SEEK_SET);
					write(external_fd, &buffer, (fileSize % 1024));
					break;
				} else {
					lseek(external_fd, count * BLOCK_SIZE, SEEK_SET);
					write(external_fd, &buffer, BLOCK_SIZE);
					totalNeededBlock--;
				}
				count++;
			}

		} else {
			//the last block in address array is triple indirect block
			int firstBlock = 0, secondBlock = 0, thirdBlock = 0, currBlock = 0;
			int firstIndex = 0, secondIndex = 0, thirdIndex = 0;

			while (1) {

				lseek(fileDescriptor,
						temp_inode.addr[ADDR_SIZE - 1] * BLOCK_SIZE
								+ firstIndex * 4, SEEK_SET);
				read(fileDescriptor, &firstBlock, 4);
				lseek(fileDescriptor, firstBlock * BLOCK_SIZE + secondIndex * 4,
						SEEK_SET);
				read(fileDescriptor, &secondBlock, 4);
				lseek(fileDescriptor, secondBlock * BLOCK_SIZE + thirdIndex * 4,
						SEEK_SET);
				read(fileDescriptor, &thirdBlock, 4);
				currBlock = thirdBlock;

				lseek(fileDescriptor, currBlock * BLOCK_SIZE, SEEK_SET);
				read(fileDescriptor, &buffer, BLOCK_SIZE);

				if (totalNeededBlock == 1) {
					lseek(external_fd, count * BLOCK_SIZE, SEEK_SET);
					write(external_fd, &buffer, (fileSize % 1024));
					break;
				} else {
					lseek(external_fd, count * BLOCK_SIZE, SEEK_SET);
					write(external_fd, &buffer, BLOCK_SIZE);
					totalNeededBlock--;
				}
				count++;

				if (thirdIndex < 255) {
					thirdIndex++;
				} else {
					thirdIndex = 0;
					secondIndex++;
				}
				if (secondIndex == 256) {
					secondIndex = 0;
					firstIndex++;
				}
			}
		}

	}

	printf(" %s is copied  to %s successfully./n", internal_path,
			external_path);
	close(external_fd);
	return;
}

void mkdirectory(char *path) {
	char *dir_name, *parentPath;
	char temp_path[1000];
	strcpy(temp_path,path);
	getFileName_and_parentPath(temp_path, "/", &dir_name, &parentPath);
	int parentInode = get_inode_number_by_path(parentPath);

	if (parentInode == -1) {
		printf("Invalid path !");
		return;
	}

	if (get_inode_number_by_parent(dir_name, parentInode) != -1) {
		printf("Directory exists !");
		return;
	}

	int dir_inode_num = allocate_inode();
	inode_type dir_inode;

//write into parent dir entry
	dir_type dir_entry;
	dir_entry.inode = dir_inode_num;
	strcpy(dir_entry.filename, dir_name);
	lseek(fileDescriptor, find_dir_entry_position_by_parent_inode(parentInode),
			SEEK_SET);
	write(fileDescriptor, &dir_entry, 16);

//initial dir_inode
	dir_inode.flags = inode_alloc_flag | dir_flag | large_file
			| dir_access_rights;
	dir_inode.nlinks = 0;
	dir_inode.uid = 0;
	dir_inode.gid = 0;
	dir_inode.size = 32;
	dir_inode.addr[0] = allocate_block_to_file();
	int i;
	for (i = 1; i < ADDR_SIZE; i++) {
		inode.addr[i] = 0;
	}

	dir_inode.actime[0] = (unsigned short) time(NULL );
	dir_inode.actime[1] = 1970;
	dir_inode.modtime[0] = (unsigned short) time(NULL );
	dir_inode.modtime[1] = 1970;

	lseek(fileDescriptor, 2 * BLOCK_SIZE + (dir_inode_num - 1) * INODE_SIZE, 0);
	write(fileDescriptor, &dir_inode, INODE_SIZE);

//write information to itself
	lseek(fileDescriptor, dir_inode.addr[0] * BLOCK_SIZE, 0);
	write(fileDescriptor, &dir_entry, 16);

	dir_type dir_entry2;
	dir_entry2.inode = parentInode;
	dir_entry2.filename[0] = '.';
	dir_entry2.filename[1] = '.';
	dir_entry2.filename[2] = '\0';
	lseek(fileDescriptor, dir_inode.addr[0] * BLOCK_SIZE + 16, 0);
	write(fileDescriptor, &dir_entry2, 16);

	return;

}

void rmFile(char* path) {
	char *file_name, *parentPath;
	char temp_path[1000];
	strcpy(temp_path, path);
	char temp_path2[1000];
    strcpy(temp_path2, path);

	getFileName_and_parentPath(temp_path2, "/", &file_name, &parentPath);

	int parent_inode_num = get_inode_number_by_path(parentPath);
	int file_inode_num = get_inode_number_by_path(temp_path);
	if ((file_inode_num == -1) || (parent_inode_num == -1)) {
		printf("Invalid path");
		return;
	}

	inode_type file_inode;
	lseek(fileDescriptor, 2 * BLOCK_SIZE + (file_inode_num - 1) * INODE_SIZE,
			SEEK_SET);
	read(fileDescriptor, &file_inode, INODE_SIZE);

//validate whether this path is directory or file
	unsigned short andnum = 24576; //060000
	andnum = file_inode.flags & andnum;
//16384 is 040000
	if (andnum == 16384) {
		printf("It is directory !");
		return;
	}

	int index;
	for (index = 0; index < ADDR_SIZE; index++) {
		if (file_inode.addr[index] == 0) {
			break;
		}
		int data_block_num = 0;

		//free top 10 single indirect blocks
		if (index < ADDR_SIZE - 1) {
			int i;
			for (i = 0; i < 256; i++) {
				lseek(fileDescriptor,
						file_inode.addr[index] * BLOCK_SIZE + i * 4, SEEK_SET);
				read(fileDescriptor, &data_block_num, 4);
				if (data_block_num == 0)
					break;
				add_block_to_free_list(data_block_num);
			}
			//free data block for  block # addr[index]
			add_block_to_free_list(file_inode.addr[index]);
		} else {
			//the last block in address array is triple indirect block
			int firstBlock = 0, secondBlock = 0, thirdBlock = 0;
			int firstIndex = 0, secondIndex = 0, thirdIndex = 0;

			while (1) {

				lseek(fileDescriptor,
						file_inode.addr[ADDR_SIZE - 1] * BLOCK_SIZE
								+ firstIndex * 4, SEEK_SET);
				read(fileDescriptor, &firstBlock, 4);
				lseek(fileDescriptor, firstBlock * BLOCK_SIZE + secondIndex * 4,
						SEEK_SET);
				read(fileDescriptor, &secondBlock, 4);
				lseek(fileDescriptor, secondBlock * BLOCK_SIZE + thirdIndex * 4,
						SEEK_SET);
				read(fileDescriptor, &thirdBlock, 4);

				if (thirdBlock == 0) {
					break;
				} else {
					add_block_to_free_list(thirdBlock);
				}
				if (thirdIndex < 255) {
					thirdIndex++;
				} else {
					thirdIndex = 0;
					add_block_to_free_list(secondBlock);
					secondIndex++;
				}
				if (secondIndex == 256) {
					secondIndex = 0;
					add_block_to_free_list(firstBlock);
					firstIndex++;
				}
			}
			//free data block for  block # addr[index]
			add_block_to_free_list(file_inode.addr[index]);
		}
	}

//free inode for this file
	inode_type temp_inode;
	temp_inode.flags = 0;
	lseek(fileDescriptor, 2 * BLOCK_SIZE + (file_inode_num - 1) * INODE_SIZE,
			SEEK_SET);
	write(fileDescriptor, &temp_inode, INODE_SIZE);

	if (superBlock.ninode < I_SIZE) {
		superBlock.inode[superBlock.ninode] = file_inode_num;
		superBlock.ninode++;
	}
	updateSuperBlock();

//remove entry in parent block
	remove_dir_entry_in_parent(parent_inode_num, file_name);
	return;
}

void rmDir(char *path) {
	char *dir_name, *parentPath;
	char temp_path[1000];
	strcpy(temp_path, path);
	char temp_child_path[1000];
	strcpy(temp_child_path, path);
	char temp_path2[1000];
	strcpy(temp_path2, path);

	getFileName_and_parentPath(temp_path2, "/", &dir_name, &parentPath);

	int parent_inode_num = get_inode_number_by_path(parentPath);
	int dir_inode_num = get_inode_number_by_path(temp_path);
	if ((dir_inode_num == -1) || (parent_inode_num == -1)) {
		printf("Invalid path");
		return;
	}

	inode_type dir_inode;
	lseek(fileDescriptor, 2 * BLOCK_SIZE + (dir_inode_num - 1) * INODE_SIZE,
			SEEK_SET);
	read(fileDescriptor, &dir_inode, INODE_SIZE);

//validate whether this path is directory or file
	unsigned short andnum = 24576; //060000
	andnum = dir_inode.flags & andnum;
//16384 is 040000
	if (andnum != 16384) {
		printf("It is not directory !");
		return;
	}

	int index;
	for (index = 2; index < 64; index++) {
		dir_type child_entry;
		lseek(fileDescriptor, dir_inode.addr[0] * BLOCK_SIZE + index * 16,
				SEEK_SET);
		read(fileDescriptor, &child_entry, 16);
		//no inode left
		if (child_entry.inode == 0) {
			break;
		}

		inode_type child_dir_inode;
		lseek(fileDescriptor,
				2 * BLOCK_SIZE + (child_entry.inode - 1) * INODE_SIZE,
				SEEK_SET);
		read(fileDescriptor, &child_dir_inode, INODE_SIZE);
		//validate whether this inode is allocated to directory
		unsigned short andnum1 = 24576; //060000
		andnum1 = child_dir_inode.flags & andnum1;
		//16384 is 040000
		if (andnum1 == 16384) {
			strcat(temp_child_path, "/");
			strcat(temp_child_path, child_entry.filename);
			//printf("child_path%s \n", temp_child_path);
			rmDir(temp_child_path);
		} else {
			strcat(temp_child_path, "/");
			strcat(temp_child_path, child_entry.filename);
			//printf("child_path%s \n", temp_child_path);
			rmFile(temp_child_path);
		}
	}
//free data block for  block # addr[index]
	add_block_to_free_list(dir_inode.addr[0]);

//free inode for this file
	inode_type temp_inode;
	temp_inode.flags = 0;
	lseek(fileDescriptor, 2 * BLOCK_SIZE + (dir_inode_num - 1) * INODE_SIZE,
			SEEK_SET);
	write(fileDescriptor, &temp_inode, INODE_SIZE);

	if (superBlock.ninode < I_SIZE) {
		superBlock.inode[superBlock.ninode] = dir_inode_num;
		superBlock.ninode++;
	}
	updateSuperBlock();

	remove_dir_entry_in_parent(parent_inode_num, dir_name);
	return;
}

void remove_dir_entry_in_parent(int parent_inode_num, char *Name) {
	dir_type temp_type;
	int addr0, i;
	lseek(fileDescriptor,
			2 * BLOCK_SIZE + (parent_inode_num - 1) * INODE_SIZE + 12,
			SEEK_SET);
//addr0 denotes block number of the block contains directory information
	read(fileDescriptor, &addr0, 4);
	for (i = 0; i < 64; i++) {
		lseek(fileDescriptor, addr0 * BLOCK_SIZE + i * 16, SEEK_SET);
		read(fileDescriptor, &temp_type, 16);
		if (strcmp(temp_type.filename, Name) == 0) {
			char fileName[14] = { 0 };
			temp_type.inode = 0;
			strcpy(temp_type.filename, fileName);
			lseek(fileDescriptor, addr0 * BLOCK_SIZE + i * 16, SEEK_SET);
			write(fileDescriptor, &temp_type, 16);
			break;
		}
	}
}

void ls() {
	dir_type dir_entry;
	char temp_path[1000];
	strcpy(temp_path, currentPath);
	int dir_blockNum, i;
	lseek(fileDescriptor,
			2 * BLOCK_SIZE
					+ (get_inode_number_by_path(temp_path) - 1) * INODE_SIZE
					+ 12, SEEK_SET);
	read(fileDescriptor, &dir_blockNum, 4);

	for (i = 2; i < 64; i++) {
		lseek(fileDescriptor, dir_blockNum * BLOCK_SIZE + i * 16, SEEK_SET);
		read(fileDescriptor, &dir_entry, 16);
		if (dir_entry.inode > 0) {
			printf("%s  ", dir_entry.filename);
		}
	}
	printf("\n");
}

void pwd() {
	printf("%s ", currentPath);
}

void cd(char *path) {
	char temp_path[1000];
	strcpy(temp_path, path);
	if (strcmp(path, "..") == 0) {
		char* parentPath = NULL;
		char* fileName = NULL;
		char temp_path1[1000];
		strcpy(temp_path1, currentPath);
		getFileName_and_parentPath(temp_path1,"/",&fileName,&parentPath);
		if(strcmp(fileName,currentPath) == 0){
			printf("No parent directory !");
			return;
		}
		currentPath = parentPath;
		return;
	}

	if (get_inode_number_by_path(temp_path) == -1) {
		printf("Invalid path !");
	} else {
		currentPath = path;
	}
}

void openExitingFileSystem() {
	char temp[1000];
	char *filePath = strtok(NULL, " ");
	strcpy(temp, filePath);

	char msg[1000][20];
	int i = 0;
	char *result = strtok(temp, "/");
//get fileSystemName
	for (i = 0; result != NULL ; i++) {
		strcpy(msg[i], result);
		result = strtok(NULL, "/");
	}
	currentPath = msg[i - 1];
	if ((fileDescriptor = open(filePath, O_RDWR, 0600)) == -1) {
		printf("\n open failed with error [%s]\n", strerror(errno));
		return;
	}

	lseek(fileDescriptor, BLOCK_SIZE, SEEK_SET);
	read(fileDescriptor, &superBlock, BLOCK_SIZE);

}

