import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { NotSupportComponent } from './not-support.component';

describe('NotSupportComponent', () => {
  let component: NotSupportComponent;
  let fixture: ComponentFixture<NotSupportComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ NotSupportComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(NotSupportComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
