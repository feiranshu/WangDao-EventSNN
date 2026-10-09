#include <linux/init.h>
#include <linux/module.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("wangdao");
MODULE_DESCRIPTION("KV260 Route C kernel module smoke test");

static int __init hello_init(void)
{
    pr_info("wangdao_hello: module loaded\n");
    return 0;
}

static void __exit hello_exit(void)
{
    pr_info("wangdao_hello: module unloaded\n");
}

module_init(hello_init);
module_exit(hello_exit);
