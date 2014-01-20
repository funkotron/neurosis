(ns neurosis.core
  (:use [nuroko.lab core charts])
  (:use [nuroko.gui visual])
  (:use [clojure.core.matrix])
  (:require [task.core :as task])
  ;; (:require [nuroko.data mnist])
  (:import [mikera.vectorz Op Ops])
  (:import [mikera.vectorz.ops ScaledLogistic Logistic Tanh])
  (:import [nuroko.coders CharCoder])
  (:import [mikera.vectorz AVector Vectorz]))

;; (ns nuroko.demo.conj)

;; some utility functions
(defn feature-img [vector]
  ((image-generator :width 8 :height 28 :colour-function weight-colour-mono) vector))  

(defn demo []

;; ============================================================
;; SCRABBLE score task

	(def scores (sorted-map "theory" 1 "playa" 0 "table" 1 "pregunta" 0 "conquistador" 0 "conquest" 1))
	
	(def score-coder (int-coder :bits 1))
	(encode score-coder 1)
	(decode score-coder (encode score-coder 0))

	(defn word-coder-helper [x] 
          (encode (int-coder :bits 8) x))

        (defn pad [total v]
          (let [n (count v)]
            (concat (take (- total n) (repeatedly (constantly 0)))
                    v)))
        
        (defn word-coder-impl [word]
          (Vectorz/create ^java.util.List (pad 160 (flatten (map concat (map word-coder-helper (map long (.getBytes word))))))))
        
        (char \a)
	(word-coder-impl "test")
	
	(def task 
	  (mapping-task scores 
	                :input-coder letter-coder
	                :output-coder score-coder))
	
	(def net 
	  (neural-network :inputs 160
	                  :outputs 1
                    :hidden-op Ops/LOGISTIC 
                    :output-op Ops/LOGISTIC
	                  :hidden-sizes [160]))
  
  (show (network-graph net :line-width 2) 
        :title "Neural Net : Scrabble")
 
  (defn language-score [net word]
    (->> word
      (word-coder-impl)
      (think net)
      (decode score-coder)))

  (language-score net "testthat")
  
  
  ;; evaluation function
  (defn evaluate-scores [net]
    (let [net (.clone net)
          words (keys scores)]
      (count (for [w words 
                   :when (= (language-score net w) (scores w))] w))))  
    
  (show (time-chart 
          [#(evaluate-scores net)] 
          :y-max 26) 
        :title "Correct letters")
   
  ;; training algorithm
  (def trainer (supervised-trainer net task :batch-size 100))
  
  (task/run 
    {:sleep 1 :repeat 1000} ;; sleep used to slow it down, otherwise trains instantly.....
    (trainer net))
   
  (language-score net "playa")
  
;; end of SCRABBLE DEMO  
  
  
  
;; ============================================================
;; MNIST digit recognistion task

  ;; ;; training data - 60,000 cases
  ;; (def data @nuroko.data.mnist/data-store)
  ;; (def labels @nuroko.data.mnist/label-store)
  ;; (def INNER_SIZE 300) 

  ;; (count data)

  ;; ;; some visualisation
  ;; ;; image display function

    
  ;; (show (map img (take 100 data)) 
  ;;       :title "First 100 digits") 

  ;; ;; we also have some labels  
  ;; (count labels)
  ;; (take 10 labels)
  
  ;; ;; ok so let's compress these images

  ;; (def compress-task (identity-task data)) 
  
  ;; (def compressor 
  ;;         (stack
  ;;     ;;(offset :length 784 :delta -0.5)
  ;;     (neural-network :inputs 784 
  ;;                           :outputs INNER_SIZE
  ;;                     :layers 1
  ;;                    ;; :max-weight-length 4.0      
  ;;                     :output-op Ops/LOGISTIC
  ;;                    ;; :dropout 0.5
  ;;                     )
  ;;     (sparsifier :length INNER_SIZE)))
  
  ;; (def decompressor 
  ;;         (stack 
  ;;     (offset :length INNER_SIZE :delta -0.5)
  ;;     (neural-network :inputs INNER_SIZE  
  ;;                           :outputs 784
  ;;                    ;; :max-weight-length 4.0
  ;;                     :output-op Ops/LOGISTIC
  ;;                     :layers 1)))
  
  ;; (def reconstructor 
  ;;   (connect compressor decompressor)) 

  ;; (defn show-reconstructions []
  ;;   (let [reconstructor (.clone reconstructor)]
  ;;     (show 
  ;;       (->> (take 100 data)
  ;;         (map (partial think reconstructor)) 
  ;;         (map img)) 
  ;;       :title "100 digits reconstructed")))
  ;; (show-reconstructions) 

  ;; (def trainer (supervised-trainer reconstructor compress-task))
  
  ;;       (task/run 
  ;;   {:sleep 1 :repeat true}
  ;;   (do (trainer reconstructor) (show-reconstructions)))
    
  ;; (task/stop-all)
 
  ;; ;; look at feature maps for 150 hidden units
  ;; (show (map feature-img (feature-maps compressor :scale 2)) :title "Feature maps") 

  
  ;; ;; now for the digit recognition
  ;; (def num-coder (class-coder 
  ;;                  :values (range 10)))
  ;;       (encode num-coder 3)
	
  ;;       (def recognition-task 
  ;;         (mapping-task 
  ;;     (apply hash-map 
  ;;            (interleave data labels)) 
  ;;           :output-coder num-coder))
  
  ;; (def recogniser
  ;;   (stack
  ;;     (offset :length INNER_SIZE :delta -0.5)
  ;;     (neural-network :inputs INNER_SIZE  
  ;;                   :output-op Ops/LOGISTIC
  ;;                         :outputs 10
  ;;                   :layers 2)))
  
  ;; (def recognition-network 
  ;;   (connect compressor recogniser))
  
  ;; (def trainer2 (supervised-trainer recognition-network 
  ;;                                   recognition-task 
  ;;                                   ;;:loss-function nuroko.module.loss.CrossEntropyLoss/INSTANCE
  ;;                                  ))

  ;; ;; test data and task - 10,000 cases
  ;; (def test-data @nuroko.data.mnist/test-data-store)
  ;; (def test-labels @nuroko.data.mnist/test-label-store)
 
  ;; (def recognition-test-task 
  ;;         (mapping-task (apply hash-map 
  ;;                       (interleave test-data test-labels)) 
  ;;                       :output-coder num-coder))
  
  ;; ;; show chart of training error (blue) and test error (red)
  ;; (show (time-chart [#(evaluate-classifier 
  ;;                       recognition-task recognition-network )
  ;;                    #(evaluate-classifier 
  ;;                       recognition-test-task recognition-network )] 
  ;;                   :y-max 1.0) 
  ;;       :title "Error rate")
  
  ;; (task/run 
  ;;   {:sleep 1 :repeat true}
  ;;   (trainer2 recognition-network :learn-rate 0.1)) 
  ;;    ;; can tune learn-rate, lower => fine tuning => able to hit better overall accuracy
    
  ;; (task/stop-all)
  
  ;; (defn recognise [image-data]
  ;;   (->> image-data
  ;;     (think recognition-network)
  ;;     (decode num-coder)))

  ;; (recognise (data 0))
  
  ;; ;; show results, errors are starred
  ;; (show (map (fn [l r] (if (= l r) l (str r "*")))
  ;;            (take 100 labels)
  ;;            (map recognise (take 100 data))) 
  ;;       :title "Recognition results") 
   
  ;; (let [rnet (.clone recognition-network)]
  ;;   (reduce 
  ;;   (fn [acc i] (if (= (test-labels i) (->> (test-data i) (think rnet) (decode num-coder))) 
  ;;                 (inc acc) acc))
  ;;   0 (range (count test-data)))) 
  
  ;; (show (class-separation-chart recognition-network (take 1000 test-data) (take 1000 test-labels)))
  
  ;; ===============================
  ;; END of DEMO
  
  
  
  ;; final view of feature maps
  (task/stop-all)
  
  ;; (show (map feature-img (feature-maps recognition-network :scale 10)) :title "Recognition maps")
  ;; (show (map feature-img (feature-maps reconstructor :scale 10)) :title "Round trip maps") 
)
