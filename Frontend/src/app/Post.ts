import { Time } from '@angular/common';

export interface Post {
    id: string;
    time: Time;
    source: string;
    sub_source: string;
    writer: string;
    text: string;
    cb_level: number;
    comment_shared_post: string;
}