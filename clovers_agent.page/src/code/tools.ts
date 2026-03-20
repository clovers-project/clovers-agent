export async function cropImageToSquare(file: File): Promise<Blob> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            if (!ctx) {
                reject(new Error("无法获取 canvas 上下文"));
                return;
            }
            // 取宽高较小值作为方形边长
            const size = Math.min(img.width, img.height);
            const offsetX = (img.width - size) / 2;
            const offsetY = (img.height - size) / 2;

            canvas.width = size;
            canvas.height = size;
            // 绘制裁剪后的方形图片
            ctx.drawImage(img, offsetX, offsetY, size, size, 0, 0, size, size);

            // 转换为 Blob
            canvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error("图片转换失败"));
                }
            }, file.type || "image/png");
        };
        img.onerror = () => reject(new Error("图片加载失败"));
        img.src = URL.createObjectURL(file);
    });
}

import { openDB } from 'idb';
const dbPromise = openDB('ChatAppDB', 1, {
    upgrade(db) {
        // 创建一个名为 "histories" 的表，以 groupId 作为键
        if (!db.objectStoreNames.contains('histories')) {
            db.createObjectStore('histories');
        }
    },
});

export const chatHistoryStorage = {
    // async set(groupId: string, html: string) {
    //     const db = await dbPromise;
    //     await db.put('histories', html, groupId);
    // },
    async append(groupId: string, html: string) {
        const db = await dbPromise;
        const records: string[] = (await db.get('histories', groupId)) || [];
        records.push(html);
        await db.put('histories', records, groupId);
    },
    async get(groupId: string): Promise<string> {
        const db = await dbPromise;
        const records = await db.get('histories', groupId) as string[] | undefined;
        if (!records || records.length === 0) return '';
        return records.join('\n');
    },
    async delete(groupId: string) {
        const db = await dbPromise;
        await db.delete('histories', groupId);
    },
    async clearAll() {
        const db = await dbPromise;
        await db.clear('histories');
    }

};