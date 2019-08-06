import { Component, OnInit } from '@angular/core';
import { PredictionsService } from './predictions.service';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-predictions',
  templateUrl: './predictions.component.html',
  styleUrls: ['./predictions.component.scss']
})
export class PredictionsComponent implements OnInit {
  explain: boolean = false;
  resultClass = null;
  classificationForm: FormGroup;
  failure = false;
  submitted = false;
  resultExplain: any;

  constructor(private predictionsService: PredictionsService, private formBuilder: FormBuilder, private _sanitizer: DomSanitizer) { }

  ngOnInit() {

  this.classificationForm = this.formBuilder.group({
    post: ['', Validators.required]
  });
  }

  toExplain(e){
    this.explain= e.target.checked;
  }
  onSubmit() {
    this.submitted = true;

    if (this.classificationForm.invalid)
      return;
    this.resultClass = null;
    this.resultExplain = false;
    this.predictionsService.getClassification(this.classificationForm.value.post, this.explain).subscribe((data) => {
      this.resultClass = data['class'];

      if('explain' in data && this.explain) {
        this.resultExplain = this._sanitizer.bypassSecurityTrustResourceUrl('data:image/png;base64,' 
                 + data['explain']);
      }
      this.failure = false;
    }, (error)=> {
      this.failure = true;
    }) 
  }

}

