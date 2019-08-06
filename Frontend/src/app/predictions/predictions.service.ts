import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class PredictionsService {

  server:string = 'http://localhost:8889';
  supported:boolean = false;
  constructor(private http: HttpClient) { }
  getClassification(post, explain) {
    return this.http.post(`${this.server}/get_classification`, {post: post, explainability: explain});
  }

}
