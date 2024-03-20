/*
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <signal.h>

static int fd = -1;  // 文件描述符
// 信号处理函数
static void sigint_handler(int sig){
    if (SIGINT != sig)
        return;
    // 解锁
    flock(fd, LOCK_UN);
    close(fd);
    printf("进程1:文件已解锁!\n");
}

int main(int argc, char *argv[]){
    if (2 != argc){
        fprintf(stderr, "usage: %s <file>\n", argv[0]);
        exit(-1);
    }
    // 打开文件
    fd = open(argv[1], O_WRONLY);
    if (-1 == fd){
        perror("open error");
        exit(-1);
    }
    // 以非阻塞方式对文件加锁（排它锁）
    if (-1 == flock(fd, LOCK_EX|LOCK_NB)){
        perror("进程1:文件加锁失败");
        exit(-1);
    }
    printf("进程1:文件加锁成功！\n");
    // 为 SIGINT 信号注册处理函数
    signal(SIGINT, sigint_handler);
    for (;;)
        sleep(1);
}
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <string.h>

int main(int argc, char *argv[]){
    char buf[100] = "hello World!";
    int fd;
    int len;
    if (2 != argc) {
        fprintf(stderr, "usage: %s <file>\n", argv[0]);
        exit(-1);
    }
    // 打开文件
    // fd = open(argv[1], O_RDWR);
    // fd = open("./tmp", O_WRONLY | O_CREAT);
    fd = open("./tmp", O_RDWR | O_CREAT);
    if (-1 == fd){
        perror("open error");
        exit(-1);
    }
    // 以非阻塞方式对文件加锁（排它锁）
    if(-1 == flock(fd, LOCK_EX | LOCK_NB))
        perror("进程2:文件加锁失败！");
    else
        printf("进程2:文件加锁成功！\n");
    // 写文件
    len = strlen(buf);
    if (0>write(fd, buf, len)){
        perror("write error");
        exit(-1);
    }
    printf("进程2:写入到文件的字符串<%s>\n", buf);
    // 将文件读写位置移动到文件头
    if (0>lseek(fd, 0x0, SEEK_SET)){
        perror("lseek error");
        exit(-1);
    }
    // 读文件
    memset(buf, 0x0, sizeof(buf)); // 清理buf
    if (0 > read(fd, buf, len)){
        perror("read error");
        exit(-1);
    }
    printf("进程2: 从文件读取的字符串<%s>\n", buf);
    // 解锁、退出
    flock(fd, LOCK_UN);
    close(fd);
    exit(0);
}
